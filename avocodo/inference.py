import os
import argparse
from omegaconf import OmegaConf
import torch
from scipy.io.wavfile import write

from pytorch_lightning import Trainer

from avocodo.meldataset import mel_spectrogram
from avocodo.meldataset import MAX_WAV_VALUE
from avocodo.meldataset import load_wav
from avocodo.meldataset import normalize
from avocodo.lightning_module import Avocodo
from avocodo.data_module import AvocodoData


h = None
device = None


def get_mel(x):
    return mel_spectrogram(
        x,
        1024,
        80,
        22050,
        256,
        1024,
        0,
        8000
    )


def inference(a, conf):
    avocodo = Avocodo.load_from_checkpoint(
        f"{a.checkpoint_path}/version_{a.version}/checkpoints/{a.checkpoint_file_id}",
        map_location='cpu'
    )
    avocodo_data = AvocodoData(conf.audio)
    avocodo_data.prepare_data()
    validation_dataloader = avocodo_data.val_dataloader()

    output_path = f'{a.output_dir}/version_{a.version}/'
    os.makedirs(output_path, exist_ok=True)

    avocodo.generator.to(a.device)
    avocodo.generator.remove_weight_norm()

    m = torch.jit.script(avocodo.generator)
    torch.jit.save(
        m,
        os.path.join(output_path, "scripted.pt")
    )

    with torch.no_grad():
        for i, batch in enumerate(validation_dataloader):
            mels, _, file_ids, _ = batch

            y_g_hat = avocodo(mels.to(a.device))

            for _y_g_hat, file_id in zip(y_g_hat, file_ids):
                audio = _y_g_hat.squeeze(0)
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')

                output_file = os.path.join(
                    output_path,
                    file_id.split('/')[-1]
                )
                print(file_id)
                write(output_file, conf.audio.sampling_rate, audio)
    print('Done inference')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='logs/Avocodo')
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--checkpoint_file_id', type=str, default='', required=True)
    parser.add_argument('--output_dir', type=str, default='generated_files')
    parser.add_argument('--script', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    a = parser.parse_args()

    conf = OmegaConf.load(os.path.join(a.checkpoint_path, f"version_{a.version}", "hparams.yaml"))
    inference(a, conf)


if __name__ == '__main__':
    main()