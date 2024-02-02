import torch
from torch import nn
from einops import rearrange

from audioldm.variational_autoencoder.modules import Encoder, Decoder
from audioldm.variational_autoencoder.distributions import DiagonalGaussianDistribution
from audioldm.hifigan.utilities import get_vocoder, vocoder_infer


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        image_key="fbank",
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
        scale_factor=1
    ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.ema_decoder = None

        self.subband = int(subband)
        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.ema_post_quant_conv = None

        self.vocoder = get_vocoder(None, "cpu")
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.scale_factor = scale_factor

    @property
    def device(self):
        return next(self.parameters()).device

    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size(-1) % self.subband == 0
        assert ch == 1

        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)

    def encode(self, x):
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.encode(x)

    def decode(self, z, use_ema=False):
        if use_ema and (not hasattr(self, 'ema_decoder') or self.ema_decoder is None):
            print("VAE does not have EMA modules, but specified use_ema. "
                  "Using the none-EMA modules instead.")
        if use_ema and hasattr(self, 'ema_decoder') and self.ema_decoder is not None:
            z = self.ema_post_quant_conv(z)
            dec = self.ema_decoder(z)
        else:
            z = self.post_quant_conv(z)
            dec = self.decoder(z)
        return self.freq_merge_subband(dec)

    def decode_first_stage(self, z, allow_grad=False, use_ema=False):
        with torch.set_grad_enabled(allow_grad):
            z = z / self.scale_factor
            return self.decode(z, use_ema)

    def decode_to_waveform(self, dec, allow_grad=False):
        dec = dec.squeeze(1).permute(0, 2, 1)
        wav_reconstruction = vocoder_infer(dec, self.vocoder, allow_grad=allow_grad)
        return wav_reconstruction

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()

        if self.flag_first_run:
            print("Latent size: ", z.size())
            self.flag_first_run = False

        return self.decode(z), posterior

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z
