import os
import importlib


CACHE_DIR = os.getenv(
    "AUDIOLDM_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/audioldm"))


def default_audioldm_config(model_name="audioldm-s-full"):    
    basic_config = {
        "wave_file_save_path": "./output",
        "id": {
            "version": "v1",
            "name": "default",
            "root": "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/AudioLDM-python/config/default/latent_diffusion.yaml",
        },
        "preprocessing": {
            "audio": {"sampling_rate": 16000, "max_wav_value": 32768},
            "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
            "mel": {
                "n_mel_channels": 64,
                "mel_fmin": 0,
                "mel_fmax": 8000,
                "freqm": 0,
                "timem": 0,
                "blur": False,
                "mean": -4.63,
                "std": 2.74,
                "target_length": 1024,
            },
        },
        "model": {
            "device": "cuda",
            "target": "audioldm.pipline.LatentDiffusion",
            "params": {
                "base_learning_rate": 5e-06,
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "first_stage_key": "fbank",
                "cond_stage_key": "waveform",
                "latent_t_size": 256,
                "latent_f_size": 16,
                "channels": 8,
                "cond_stage_trainable": True,
                "conditioning_key": "film",
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audioldm.latent_diffusion.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "extra_film_condition_dim": 512,
                        "extra_film_use_concat": True,
                        "in_channels": 8,
                        "out_channels": 8,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "use_spatial_transformer": True,
                    },
                },
                "first_stage_config": {
                    "base_learning_rate": 4.5e-05,
                    "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
                    "params": {
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 8,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "z_channels": 8,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.0,
                        },
                    },
                },
                "cond_stage_config": {
                    "target": "audioldm.clap.encoders.CLAPAudioEmbeddingClassifierFreev2",
                    "params": {
                        "key": "waveform",
                        "sampling_rate": 16000,
                        "embed_mode": "audio",
                        "unconditional_prob": 0.1,
                    },
                },
            },
        },
    }
    
    if("-l-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["model_channels"] = 256
        basic_config["model"]["params"]["unet_config"]["params"]["num_head_channels"] = 64
    elif("-m-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["model_channels"] = 192
        basic_config["model"]["params"]["cond_stage_config"]["params"]["amodel"] = "HTSAT-base" # This model use a larger HTAST
        
    return basic_config


def get_metadata():
    return {
        "audioldm-s-full": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-s-full.ckpt",
            ),
            "url": "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1",
        },
        "audioldm-l-full": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-l-full.ckpt",
            ),
            "url": "https://zenodo.org/record/7698295/files/audioldm-full-l.ckpt?download=1",
        },
        "audioldm-s-full-v2": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-s-full-v2.ckpt",
            ),
            "url": "https://zenodo.org/record/7698295/files/audioldm-full-s-v2.ckpt?download=1",
        },
        "audioldm-m-text-ft": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-m-text-ft.ckpt",
            ),
            "url": "https://zenodo.org/record/7813012/files/audioldm-m-text-ft.ckpt?download=1",
        },
        "audioldm-s-text-ft": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-s-text-ft.ckpt",
            ),
            "url": "https://zenodo.org/record/7813012/files/audioldm-s-text-ft.ckpt?download=1",
        },
        "audioldm-m-full": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-m-full.ckpt",
            ),
            "url": "https://zenodo.org/record/7813012/files/audioldm-m-full.ckpt?download=1",
        },
    }


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
