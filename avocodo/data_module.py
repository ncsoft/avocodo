from dataclasses import dataclass
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from avocodo.meldataset import MelDataset
from avocodo.meldataset import get_dataset_filelist


@dataclass
class AvocodoDataConfig:
    segment_size: int
    num_mels: int
    num_freq: int
    sampling_rate: int
    n_fft: int
    hop_size: int
    win_size: int
    fmin: int
    fmax: int
    batch_size: int
    num_workers: int

    fine_tuning: bool
    base_mels_path: str

    input_wavs_dir: str
    input_mels_dir: str
    input_training_file: str
    input_validation_file: str


class AvocodoData(LightningDataModule):
    def __init__(self, h: AvocodoDataConfig):
        super().__init__()
        self.save_hyperparameters(h)

    def prepare_data(self):
        '''
            download and prepare data
        '''
        self.training_filelist, self.validation_filelist = get_dataset_filelist(
            self.hparams.input_wavs_dir,
            self.hparams.input_training_file,
            self.hparams.input_validation_file
        )

    def setup(self, stage=None):
        self.trainset = MelDataset(
            self.training_filelist,
            self.hparams.segment_size,
            self.hparams.n_fft,
            self.hparams.num_mels,
            self.hparams.hop_size,
            self.hparams.win_size,
            self.hparams.sampling_rate,
            self.hparams.fmin,
            self.hparams.fmax,
            n_cache_reuse=0,
            fmax_loss=self.hparams.fmax_for_loss,
            fine_tuning=self.hparams.fine_tuning,
            base_mels_path=self.hparams.input_mels_dir
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            drop_last=True
        )

    @rank_zero_only
    def val_dataloader(self):
        validset = MelDataset(
            self.validation_filelist,
            self.hparams.segment_size,
            self.hparams.n_fft,
            self.hparams.num_mels,
            self.hparams.hop_size,
            self.hparams.win_size,
            self.hparams.sampling_rate,
            self.hparams.fmin,
            self.hparams.fmax,
            False,
            False,
            n_cache_reuse=0,
            fmax_loss=self.hparams.fmax_for_loss,
            fine_tuning=self.hparams.fine_tuning,
            base_mels_path=self.hparams.input_mels_dir
        )
        return DataLoader(validset, num_workers=self.hparams.num_workers, shuffle=False,
                          sampler=None,
                          batch_size=1,
                          pin_memory=True,
                          drop_last=True)