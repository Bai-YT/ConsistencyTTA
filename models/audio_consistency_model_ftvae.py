from copy import deepcopy
from typing import Any, Mapping

from accelerate.logging import get_logger
from models.audio_consistency_model import AudioLCM
from tools.train_utils import do_ema_update

logger = get_logger(__name__, log_level="INFO")


class AudioLCM_FTVAE(AudioLCM):

    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
        use_edm=False,
        use_karras=False,
        use_lora=False,
        target_ema_decay=.95,
        ema_decay=.999,
        num_diffusion_steps=18,
        teacher_guidance_scale=1,
        vae=None,
        loss_type='clap'
    ):
        assert loss_type == 'clap'
        super().__init__(
            text_encoder_name=text_encoder_name,
            scheduler_name=scheduler_name,
            unet_model_name=unet_model_name,
            unet_model_config_path=unet_model_config_path,
            snr_gamma=snr_gamma,
            freeze_text_encoder=freeze_text_encoder,
            uncondition=uncondition,
            use_edm=use_edm,
            use_karras=use_karras,
            use_lora=use_lora,
            target_ema_decay=target_ema_decay,
            ema_decay=ema_decay,
            num_diffusion_steps=num_diffusion_steps,
            teacher_guidance_scale=teacher_guidance_scale,
            vae=vae,
            loss_type=loss_type
        )

        # EMA VAE decoder
        self.ema_vae_decoder = deepcopy(self.vae.decoder)
        self.ema_vae_decoder.requires_grad_(False)
        self.ema_vae_decoder.eval()
        self.ema_vae_pqconv = deepcopy(self.vae.post_quant_conv)
        self.ema_vae_pqconv.requires_grad_(False)

        # Enable VAE training
        self.vae.decoder.train(self.training)
        self.vae.decoder.requires_grad_(True)
        self.vae.post_quant_conv.requires_grad_(True)

        self.vae.ema_decoder = self.ema_vae_decoder
        self.vae.ema_post_quant_conv = self.ema_vae_pqconv

        logger.info("Fine-tuning the VAE weights in addition to the U-Net.")

    def load_pretrained(self, state_dict: Mapping[str, Any], strict: bool = True):
        return_info = super().load_pretrained(state_dict, strict)

        # Additionally, load the VAE decoder weights
        for module, name in zip(
            [self.vae.decoder, self.vae.post_quant_conv, self.ema_vae_decoder, self.ema_vae_pqconv],
            ['vae.decoder', 'vae.post_quant_conv', 'ema_vae_decoder', 'ema_vae_pqconv']
        ):
            name_offset = len(name) + 1
            module_sd = {}
            for key, val in state_dict.items():
                if name in key:
                    if 'loss' in key:
                        new_key = key[5:]
                        if new_key[name_offset:] not in module_sd.keys():
                            module_sd[new_key[name_offset:]] = val
                    else:
                        module_sd[key[name_offset:]] = val
            module.load_state_dict(module_sd)

        self.vae.ema_decoder = self.ema_vae_decoder
        self.vae.ema_post_quant_conv = self.ema_vae_pqconv
        return return_info

    def train(self, mode: bool = True):
        super().train(mode)
        self.ema_vae_decoder.eval()
        self.vae.decoder.train(mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def update_ema(self):
        super().update_ema()
        do_ema_update(
            source_model=self.vae.decoder,
            shadow_models=[self.ema_vae_decoder],
            decay_consts=[self.ema_decay]
        )
        do_ema_update(
            source_model=self.vae.post_quant_conv,
            shadow_models=[self.ema_vae_pqconv],
            decay_consts=[self.ema_decay]
        )

    def check_eval_mode(self):
        super(AudioLCM, self).check_eval_mode()  # Call grandparent's check_eval_mode

        for model, name in zip(
            [self.student_target_unet, self.ema_vae_decoder, self.vae.vocoder],
            ['student_target_unet', 'ema_vae_decoder', 'vae.vocoder']
        ):
            assert model.training == False, f"The {name} is not in eval mode."
            for param in model.parameters():
                assert param.requires_grad == False, f"The {name} is not frozen."

        for param in self.ema_vae_pqconv.parameters():
            assert param.requires_grad == False, "The ema_vae_pqconv is not frozen."
