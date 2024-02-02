import torch

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata


def build_pretrained_models(name):
    checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    # Only load the first_stage_model
    vae_state_dict = {
        k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k
    }

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],  # default 1024
        config["preprocessing"]["stft"]["hop_length"],  # default 160
        config["preprocessing"]["stft"]["win_length"],  # default 1024
        config["preprocessing"]["mel"]["n_mel_channels"],  # default 64
        config["preprocessing"]["audio"]["sampling_rate"],  # default 16000
        config["preprocessing"]["mel"]["mel_fmin"],  # default 0
        config["preprocessing"]["mel"]["mel_fmax"],  # default 8000
    )

    vae.eval(); vae.requires_grad_(False)
    fn_STFT.eval(); fn_STFT.requires_grad_(False)
    return vae, fn_STFT
