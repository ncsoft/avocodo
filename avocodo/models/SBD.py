from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm
from typing import List

from avocodo.pqmf import PQMF
from avocodo.models.utils import get_padding


class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(MDC, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_k,
                    dilation=_d,
                    padding=get_padding(_k, _d)
                ))
            )
        self.post_conv = norm_f(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides,
            padding=get_padding(_k, _d)
        ))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        _out = None
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)
            _x = F.leaky_relu(_x, 0.2)
            if _out is None:
                _out = _x
            else:
                _out = torch.cat([_out, _x], axis=-1)
        x = torch.sum(_out, dim=-1)
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)  # @@

        return x


class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim,
        strides,
        filters,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        for _s, _f, _k, _d in zip(
            strides,
            filters_in_out,
            kernel_size,
            dilations
        ):
            self.convs.append(MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
                use_spectral_norm=use_spectral_norm
            ))
        self.post_conv = norm_f(Conv1d(
            in_channels=_f[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=3 // 2
        ))  # @@

    def forward(self, x):
        fmap = []
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)
        x = self.post_conv(x)  # @@

        return x, fmap


class MDCDConfig:
    def __init__(self, h):
        self.pqmf_params = h.pqmf_config["sbd"]
        self.f_pqmf_params = h.pqmf_config["fsbd"]
        self.filters = h.sbd_filters
        self.kernel_sizes = h.sbd_kernel_sizes
        self.dilations = h.sbd_dilations
        self.strides = h.sbd_strides
        self.band_ranges = h.sbd_band_ranges
        self.transpose = h.sbd_transpose
        self.segment_size = h.segment_size


class SBD(torch.nn.Module):
    def __init__(self, h, use_spectral_norm=False):
        super(SBD, self).__init__()
        self.config = MDCDConfig(h)
        self.pqmf = PQMF(
            *self.config.pqmf_params
        )
        if True in h.sbd_transpose:
            self.f_pqmf = PQMF(
                *self.config.f_pqmf_params
            )
        else:
            self.f_pqmf = None

        self.discriminators = torch.nn.ModuleList()

        for _f, _k, _d, _s, _br, _tr in zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.strides,
            self.config.band_ranges,
            self.config.transpose
        ):
            if _tr:
                segment_dim = self.config.segment_size // _br[1] - _br[0]
            else:
                segment_dim = _br[1] - _br[0]

            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=_f,
                kernel_size=_k,
                dilations=_d,
                strides=_s,
                use_spectral_norm=use_spectral_norm
            ))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        if self.f_pqmf is not None:
            y_in_f = self.f_pqmf.analysis(y)
            y_hat_in_f = self.f_pqmf.analysis(y_hat)

        for d, br, tr in zip(
            self.discriminators,
            self.config.band_ranges,
            self.config.transpose
        ):
            if tr:
                _y_in = y_in_f[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0]:br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
                _y_hat_in = torch.transpose(_y_hat_in, 1, 2)
            else:
                _y_in = y_in[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in[:, br[0]:br[1], :]
            y_d_r, fmap_r = d(_y_in)
            y_d_g, fmap_g = d(_y_hat_in)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs