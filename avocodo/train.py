import os
import argparse

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from avocodo.data_module import AvocodoData
from avocodo.lightning_module import Avocodo


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file',
                        default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file',
                        default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--config', default='avocodo/configs/avocodo_v1.json')
    parser.add_argument('--training_epochs', default=5000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--resume_checkpoint_path', default=None, type=str, help="Path to the checkpoint to resume training/finetune on")

    a = parser.parse_args()
    OmegaConf.register_new_resolver(
        "from_args", lambda x: getattr(a, x)
    )
    OmegaConf.register_new_resolver(
        "dir", lambda base_dir, string: os.path.join(base_dir, string)
    )
    conf = OmegaConf.load(a.config)
    OmegaConf.resolve(conf)

    dm = AvocodoData(conf.data)
    model = Avocodo(conf.model)
    if a.resume_checkpoint_path is not None:
        model.load_from_checkpoint(a.resume_checkpoint_path)

    limit_train_batches = 1.0
    limit_val_batches = 1.0
    log_every_n_steps = 50
    max_epochs = conf.model.train.training_epochs

    trainer = Trainer(
        gpus=1,
        max_epochs=max_epochs,
        callbacks=[
            RichProgressBar(
                refresh_rate=1,
                theme=RichProgressBarTheme(
                    description="#AF81EB",
                    progress_bar="#8BE9FE",
                    progress_bar_finished="#8BE9FE",
                    progress_bar_pulse="#1363DF",
                    batch_progress="#AF81EB",
                    time="#1363DF",
                    processing_speed="#1363DF",
                    metrics="#9BF9FE",
                )
            )
        ],
        logger=TensorBoardLogger("logs", name="Avocodo"),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps
    )
    trainer.fit(model, dm)
