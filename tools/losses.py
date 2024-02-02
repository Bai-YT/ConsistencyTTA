import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torchaudio.functional import resample

import laion_clap


def reduce(loss, reduction):
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'instance':
        return loss
    else:
        raise ValueError("Unknown loss reduction option.")


class MSELoss(Module):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'instance') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self, input: Tensor, target: Tensor, gt_wav: Tensor, gt_text: Tensor
    ) -> Tensor:
        raw_loss = F.mse_loss(input.float(), target.float(), reduction='none')
        instance_loss = raw_loss.mean(dim=list(range(1, len(raw_loss.shape))))
        return reduce(instance_loss, self.reduction)


class MelLoss(Module):
    __constants__ = ['reduction']

    def __init__(
        self, vae: Module, reduction: str = 'instance', mse_weight = .7, mel_weight=.3
    ) -> None:
        super().__init__()
        self.vae = vae
        self.reduction = reduction
        self.mse_weight, self.mel_weight = mse_weight, mel_weight

    def forward(
        self, input: Tensor, target: Tensor, gt_wav: Tensor, gt_text: Tensor
    ) -> Tensor:
        input_mel = self.vae.decode_first_stage(input.float(), allow_grad=True)
        target_mel = self.vae.decode_first_stage(target.float(), allow_grad=True)

        # Mel loss component
        raw_loss_1 = F.mse_loss(
            input_mel.float(), target_mel.float(), reduction='none'
        ) * self.mel_weight
        instance_loss_1 = raw_loss_1.mean(dim=list(range(1, len(raw_loss_1.shape))))

        # Latent MSE loss component
        raw_loss_2 = F.mse_loss(
            input.float(), target.float(), reduction='none'
        ) * self.mse_weight
        instance_loss_2 = raw_loss_2.mean(dim=list(range(1, len(raw_loss_2.shape))))
        return reduce(instance_loss_1 + instance_loss_2, self.reduction)


class SpectralConvergengeLoss(Module):
    """
    Spectral convergence loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L36
    """

    def __init__(self, reduction='instance'):
        """Initilize spectral convergence loss module."""
        super().__init__()
        self.reduction = reduction

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor):
                Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor):
                Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        numer = torch.norm(y_mag - x_mag, p="fro", dim=list(range(1, len(y_mag.shape))))
        denom = torch.norm(y_mag, p="fro", dim=list(range(1, len(y_mag.shape))))
        instance_loss = numer / denom
        return reduce(instance_loss, self.reduction)


class LogSTFTMagnitudeLoss(Module):
    """
    Log STFT magnitude loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L54
    """

    def __init__(self, reduction='instance'):
        """Initilize los STFT magnitude loss module."""
        super().__init__()
        self.reduction = reduction

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): 
                Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): 
                Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        raw_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag), reduction='none')
        instance_loss = raw_loss.mean(dim=list(range(1, len(raw_loss.shape))))
        return reduce(instance_loss, self.reduction)


class STFTLoss(Module):
    """
    STFT loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L72
    """

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600,
        window="hann_window", reduction='instance'
    ):
        """Initialize STFT loss module."""
        super().__init__()

        # STFT parameters
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

        self.spectral_convergenge_loss = SpectralConvergengeLoss(reduction)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(reduction)

    def stft(self, x):
        """
        Perform STFT and convert to magnitude spectrogram.
        Adapted from https://github.com/facebookresearch/denoiser/blob/
        8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L17

        Args:
            x (Tensor): Input signal tensor (B, T).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.
        Returns:
            Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
        """
        x_stft = torch.view_as_real(torch.stft(
            x.double(), self.fft_size, self.shift_size, self.win_length,
            self.window, return_complex=True
        ))
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        # Clamping is needed to avoid nan or inf
        mag = real ** 2 + imag ** 2
        return torch.clamp(mag, min=1e-8).sqrt().transpose(2, 1).float()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, T).
            target (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        input_stft = self.stft(input)
        target_stft = self.stft(target)
        sc_loss = self.spectral_convergenge_loss(input_stft, target_stft)
        mag_loss = self.log_stft_magnitude_loss(input_stft, target_stft)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(Module):
    """
    Multi resolution STFT loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L102
    """
    __constants__ = ['reduction']

    def __init__(
        self, vae: Module, reduction: str = 'instance', fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window",
        factor_sc=0.2, factor_mag=0.2, factor_mse=1
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            vae (nn.Module):    TANGO VAE.
            reduction (string): Reduction mode.
            fft_sizes (list):   List of FFT sizes.
            hop_sizes (list):   List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str):       Window function type.
            factor (float):     A balancing factor across different losses.
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.vae = vae
        self.reduction = reduction

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, reduction)]

        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_mse = factor_mse

    def forward(
        self, input: Tensor, target: Tensor, gt_wav: Tensor, gt_text: Tensor
    ) -> Tensor:
        """Calculate forward propagation.
        Args:
            input (Tensor):  Predicted latent representation.
            target (Tensor): Target latent representation.
        Returns:
            Tensor: Weighted STFT and MSE loss.
        """
        raw_mse_loss = F.mse_loss(input.float(), target.float(), reduction='none')
        instance_mse_loss = raw_mse_loss.mean(dim=list(range(1, len(raw_mse_loss.shape))))
        mse_loss = reduce(instance_mse_loss, self.reduction)

        input_mel = self.vae.decode_first_stage(input.float(), allow_grad=True)
        input_wav = self.vae.decode_to_waveform(input_mel.float(), allow_grad=True)
        input_wav = input_wav[:, :int(self.sr * 10)]  # Truncate to 10 seconds
        target_mel = self.vae.decode_first_stage(target.float(), allow_grad=True)
        target_wav = self.vae.decode_to_waveform(target_mel.float(), allow_grad=True)
        target_wav = target_wav[:, :int(self.sr * 10)]  # Truncate to 10 seconds

        sc_loss, mag_loss = 0., 0.

        for loss in self.stft_losses:
            sc_l, mag_l = loss(input_wav, target_wav)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_mse * mse_loss + \
            self.factor_mag * mag_loss + self.factor_sc * sc_loss


class CLAPLoss(Module):
    def __init__(
        self, vae: Module, reduction: str = 'instance',
        mse_weight: float = 1., clap_weight: float = 1.
    ):
        super().__init__()

        self.vae = vae
        self.reduction = reduction
        self.sr = 16000

        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        self.clap.load_ckpt('ckpt/music_audioset_epoch_15_esc_90.14.pt')
        self.clap.eval()
        self.clap.requires_grad_(False)

        self.mse_weight = mse_weight
        self.clap_weight = clap_weight
    
    def forward(
        self, input: Tensor, target: Tensor, gt_wav: Tensor, captions: Tensor,
        use_ema: bool = False
    ) -> Tensor:
        """Calculate forward propagation.
        Args:
            input (Tensor):  Predicted latent representation.
            target (Tensor): Predicted latent representation.
            groundtruth (Tensor): Ground truth audio waveform.
        Returns:
            Tensor: Weight CLAP and MSE loss.
        """
        raw_mse_loss = F.mse_loss(input.float(), target.float(), reduction='none')
        instance_mse_loss = raw_mse_loss.mean(dim=list(range(1, len(raw_mse_loss.shape))))
        mse_loss = reduce(instance_mse_loss, self.reduction)

        input_mel = self.vae.decode_first_stage(
            input.float(), allow_grad=True, use_ema=use_ema
        )
        input_wav = self.vae.decode_to_waveform(input_mel.float(), allow_grad=True)
        input_wav = input_wav[:, :int(self.sr * 10)]

        # Truncate to 10 seconds and resample to 48kHz for CLAP
        input_wav, gt_wav = tuple(resample(
            wav[:, :int(self.sr * 10)], orig_freq=self.sr, new_freq=48000,
            lowpass_filter_width=64, rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492,
        ) for wav in (input_wav, gt_wav))

        input_feat = self.clap.get_audio_embedding_from_data(input_wav, use_tensor=True)
        gt_wav_feat = self.clap.get_audio_embedding_from_data(gt_wav, use_tensor=True)
        caption_feat = self.clap.get_text_embedding(captions, use_tensor=True)

        gen_text_similarity = F.cosine_similarity(input_feat, caption_feat, dim=1)
        gen_gt_similarity = F.cosine_similarity(input_feat, gt_wav_feat, dim=1)

        instance_loss = self.mse_weight * mse_loss + \
            self.clap_weight * (2 - gen_text_similarity - gen_gt_similarity)
        return reduce(instance_loss, self.reduction)
