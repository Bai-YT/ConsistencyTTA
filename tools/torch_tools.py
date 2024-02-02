import os
import random
import itertools
import numpy as np
import soundfile as sf
import librosa, resampy
import torch

from tools.mix import mix


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)

    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length)
        waveform = torch.cat([waveform, pad_wav.to(waveform.device)])
        return waveform


def _pad_spec(fbank, target_length=1024):

    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]

    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]

    return fbank


def read_wav_file(filename, segment_length, target_sr=16000):

    wav, orig_sr = sf.read(filename)
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav.T)

    # Resample
    if not isinstance(target_sr, list):
        target_sr = [target_sr]
    cur_sr = orig_sr
    for tar_sr in target_sr:
        if cur_sr != tar_sr:
            wav = resampy.resample(wav, cur_sr, tar_sr, filter="kaiser_best")
            cur_sr = tar_sr

    wav = torch.from_numpy(wav)
    wav = wav - wav.mean()
    wav = wav / (wav.abs().max() + 1e-8) / 2
    wav = pad_wav(wav, segment_length).unsqueeze(0)
    wav = wav / (wav.abs().max() + 1e-8) / 2

    return wav.float()


def get_mel_from_wav(audio, _stft):
    audio = torch.nan_to_num(torch.clip(audio, -1, 1))
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    return melspec, log_magnitudes_stft, energy


def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ""


def mix_wavs_and_captions(wave1, wave2, caption1, caption2):
    mixed_sound = mix(wave1, wave2, 0.5, 16000).reshape(1, -1)
    mixed_caption = f"{caption1} and {uncapitalize(caption2)}"
    return mixed_sound, mixed_caption


def augment(waveforms, texts, num_items=None):
    """ num_items is the number of augmented examples per batch
    """
    if num_items == None:
        num_items = len(texts) // 2
    if torch.is_tensor(waveforms):
        waveforms = waveforms.cpu().numpy()

    combinations = list(itertools.combinations(list(range(len(texts))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]

    mixed_sound_list, mixed_caption_list = [], []
    for (i, j) in selected_combinations:
        mixed_sound = mix(waveforms[i, :], waveforms[j, :], 0.5, 16000)
        mixed_caption = f"{texts[i]} and {uncapitalize(texts[j])}"
        mixed_sound_list.append(mixed_sound.reshape(1, -1))
        mixed_caption_list.append(mixed_caption)

    mixed_waveforms = torch.tensor(np.concatenate(mixed_sound_list, 0))
    mixed_waveforms = mixed_waveforms / mixed_waveforms.abs().max() / 2

    return mixed_waveforms, mixed_caption_list


def wav_to_fbank(waveforms, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveforms, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank = _pad_spec(fbank, target_length)
    log_magnitudes_stft = _pad_spec(log_magnitudes_stft, target_length)
    return fbank, log_magnitudes_stft
