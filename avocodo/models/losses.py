# -*- coding: utf-8 -*-

# Copyright 2020 Jungil Kong
#  MIT License (https://opensource.org/licenses/MIT)

"""."""
'''
Copied from https://github.com/jik876/hifi-gan/blob/master/models.py
'''

import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    losses = []
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            _loss = torch.mean(torch.abs(rl - gl))
            loss += _loss
        losses.append(_loss)

    return loss*2, losses


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses