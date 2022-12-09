---------
🥑 Avocodo: Generative Adversarial Network for Artifact-Free Vocoder
---------

**Accepted for publication in the 37th AAAI conference on artificial intelligence.**

.. image:: https://img.shields.io/badge/arXiv-2211.04610-red.svg?style=plastic
   :target: https://arxiv.org/abs/2206.13404

.. image:: https://img.shields.io/badge/Sample_Page-Avocodo-blue.svg?style=plastic
   :target: https://nc-ai.github.io/speech/publications/Avocodo/index.html

.. image:: https://img.shields.io/badge/NC_SpeechAI-publications-brightgreen.svg?style=plastic
   :target: https://nc-ai.github.io/speech/


In our `paper <https://arxiv.org/abs/2206.13404>`_, we proposed ``Avocodo``.
We provide our implementation as an open source in this repository.

**Abstract :** Neural vocoders based on the generative adversarial neural network (GAN) have been widely used due to their fast inference speed and lightweight networks while generating high-quality speech waveforms. Since the perceptually important speech components are primarily concentrated in the low-frequency bands, most GAN-based vocoders perform multi-scale analysis that evaluates downsampled speech waveforms. This multi-scale analysis helps the generator improve speech intelligibility. However, in preliminary experiments, we discovered that the multi-scale analysis which focuses on the low-frequency bands causes unintended artifacts, e.g., aliasing and imaging artifacts, which degrade the synthesized speech waveform quality. Therefore, in this paper, we investigate the relationship between these artifacts and GAN-based vocoders and propose a GAN-based vocoder, called Avocodo, that allows the synthesis of high-fidelity speech with reduced artifacts. We introduce two kinds of discriminators to evaluate speech waveforms in various perspectives: a collaborative multi-band discriminator and a sub-band discriminator. We also utilize a pseudo quadrature mirror filter bank to obtain downsampled multi-band speech waveforms while avoiding aliasing. According to experimental resutls, Avocodo outperforms baseline GAN-based vocoders, both objectviely and subjectively, while reproducing speech with fewer artifacts.

Pre-requisites
===============

1. Install pyenv
  - `pyenv <https://github.com/pyenv/pyenv>`_
  - `pyenv automatic installer <https://github.com/pyenv/pyenv-installer>`_ (recommended)
2. Clone this repository
3. Setup virtual environment and install python requirements. Please refer pyproject.toml
  .. code-block::

    pyenv install 3.8.11
    pyenv virtualenv 3.8.11 avocodo
    pyenv local avocodo

    pip install wheel
    pip install poetry

    poetry install
4. Download and extract the `LJ Speech dataset <https://keithito.com/LJ-Speech-Dataset>`_. And move all wav files to LJSpeech-1.1/wavs


Training
===============
  .. code-block::

    python avocodo/train.py --config avocodo/configs/avocodo_v1.json

Inference
===============
  .. code-block::

    python avocodo/inference.py --version ${version} --checkpoint_file_id ${checkpoint_file_id}

Reference
===============
We referred to below repositories to make this project.

  `HiFi-GAN <https://github.com/jik876/hifi-gan>`_

  `Parallel-WaveGAN <https://github.com/kan-bayashi/ParallelWaveGAN>`_