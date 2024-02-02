from copy import deepcopy
from typing import Any, Mapping

import torch
import torch.nn as nn

from transformers import CLIPTokenizer, AutoTokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel
from accelerate.logging import get_logger

from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DConditionGuidedModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from tools.train_utils import do_ema_update

logger = get_logger(__name__, log_level="INFO")


class AudioDistilledModel(nn.Module):

    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        use_lora=False,
        ema_decay=.999,
        teacher_guidance_scale=3,
        **kwargs
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, \
            "Either UNet pretrain model name or a config file path is required"

        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.freeze_text_encoder = freeze_text_encoder
        self.use_lora = use_lora
        self.ema_decay = ema_decay
        self.snr_gamma = snr_gamma  # Default it 5 for TANGO
        logger.info(f"SNR gamma: {self.snr_gamma}")
        self.noise_scheduler = None

        # If -1, then use variable guidance scale following the distribution Unif(0, 6)
        self.teacher_guidance_scale = teacher_guidance_scale
        self.max_rand_guidance_scale = 6
        var_string = f"variable (max {self.max_rand_guidance_scale})"
        logger.info(
            "Teacher guidance scale: "
            f"{var_string if teacher_guidance_scale == -1 else teacher_guidance_scale}"
        )

        # Initialize the teacher diffusion U-Net 
        self.set_from = "random"
        unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
        self.teacher_unet = UNet2DConditionModel.from_config(
            unet_config, subfolder="unet"
        )
        student_model_class = UNet2DConditionGuidedModel if \
            self.teacher_guidance_scale == -1 else UNet2DConditionModel
        self.student_unet = student_model_class.from_config(
            unet_config, subfolder="unet"
        )

        # Initialize the student diffusion U-Net with the teacher weights
        load_info = self.student_unet.load_state_dict(
            self.teacher_unet.state_dict(), strict=False
        )
        # print(f"Keys that are not loaded: {load_info.missing_keys}")
        assert len(load_info.unexpected_keys) == 0, \
            f"Redundant keys in state_dict: {load_info.unexpected_keys}"
        # Initialize the EMA copy of the student diffusion U-Net
        self.student_ema_unet = deepcopy(self.student_unet)

        self.lightweight = 'light' in unet_model_config_path
        logger.info(f"Using the lightweight setting: {self.lightweight}")

        # Set eval mode
        for model in [self.teacher_unet, self.student_ema_unet]:
            model.eval()
            model.requires_grad_(False)

        # Initialize text encoder (default is FLAN-T5)
        if "stable-diffusion" in self.text_encoder_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.text_encoder_name, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.text_encoder_name, subfolder="text_encoder"
            )
        elif "t5" in self.text_encoder_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)

        if self.freeze_text_encoder:
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
            logger.info("Text encoder is frozen.")

    @property
    def device(self):  # Need this to enable single GPU training
        return self.text_encoder.device

    @property
    def use_teacher_cf_guidance(self):
        return self.teacher_guidance_scale == -1 or self.teacher_guidance_scale > 1.

    def setup_lora(self):
        """ Set up Low-Rank Adaptation (LoRA)
        """
        logger.info("Setting up low-rank adaptation.")
        self.student_unet.requires_grad_(False)
        lora_attn_procs = {}

        for name in self.student_unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else 
                self.student_unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.student_unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(
                    reversed(self.student_unet.config.block_out_channels)
                )[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.student_unet.config.block_out_channels[block_id]

            if self.lightweight:
                hidden_size = hidden_size * 255 // 256
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.student_unet.set_attn_processor(lora_attn_procs)

    def load_state_dict_from_tango(self, **kwargs):
        raise NotImplementedError

    def load_pretrained(self, state_dict: Mapping[str, Any], strict: bool = True):
        # Placeholder function
        return self.load_state_dict(state_dict, strict)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        self.teacher_unet.eval()
        self.student_ema_unet.eval()
        return self

    def eval(self):
        return self.train(mode=False)

    def compute_snr(self, timesteps):
        """ Computes the signal-noise ratio (SNR) as per 
            https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/
            521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/
            gaussian_diffusion.py#L847-L849
            Compatible with the DDPM scheduler.
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors. Adapted from L1026 of
        # https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/
        # 521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = (
            sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        )
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        return (alpha / sigma) ** 2

    def encode_text(self, prompt, max_length=None, padding=True):
        device = self.text_encoder.device
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        batch = self.tokenizer(
            prompt, max_length=max_length, padding=padding,
            truncation=True, return_tensors="pt"
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                prompt_embeds = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        bool_prompt_mask = (attention_mask == 1).to(device)  # Convert to boolean
        return prompt_embeds, bool_prompt_mask

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        # get conditional embeddings
        cond_prompt_embeds, cond_prompt_mask = self.encode_text(prompt)
        cond_prompt_embeds = cond_prompt_embeds.repeat_interleave(
            num_samples_per_prompt, 0
        )
        cond_prompt_mask = cond_prompt_mask.repeat_interleave(
            num_samples_per_prompt, 0
        )

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)
        negative_prompt_embeds, uncond_prompt_mask = self.encode_text(
            uncond_tokens, max_length=cond_prompt_embeds.shape[1], padding="max_length"
        )
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
            num_samples_per_prompt, 0
        )
        uncond_prompt_mask = uncond_prompt_mask.repeat_interleave(
            num_samples_per_prompt, 0
        )

        """ For classifier free guidance, we need to do two forward passes.
            We concatenate the unconditional and text embeddings into a single batch 
            to avoid doing two forward passes
        """
        prompt_embeds = torch.cat([negative_prompt_embeds, cond_prompt_embeds])
        prompt_mask = torch.cat([uncond_prompt_mask, cond_prompt_mask])

        return prompt_embeds, prompt_mask, cond_prompt_embeds, cond_prompt_mask

    def get_prompt_embeds(self, prompt, use_cf_guidance, num_samples_per_prompt=1):
        """ Return: 
            prompt_embeds of cond+uncond, prompt_mask_cf of cond+uncond, 
            prompt_embeds of cond only, prompt_mask of cond only
        """
        if use_cf_guidance:  # Use classifier-free guidance
            return self.encode_text_classifier_free(prompt, num_samples_per_prompt)

        else:  # Do not use guidance
            prompt_embeds, prompt_mask = self.encode_text(prompt)
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            prompt_mask = prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

        return prompt_embeds, prompt_mask, prompt_embeds, prompt_mask

    def update_ema(self):
        assert self.training, "EMA update should only be called during training"
        do_ema_update(
            source_model=self.student_unet,
            shadow_models=[self.student_ema_unet],
            decay_consts=[self.ema_decay]
        )

    def check_eval_mode(self):
        # Check frozen parameters and eval mode
        models = [self.text_encoder, self.teacher_unet, self.student_ema_unet]
        names = ["text_encoder", "teacher_unet", "student_ema_unet"]

        for model, name in zip(
            models if self.freeze_text_encoder else models[1:],
            names if self.freeze_text_encoder else names[1:]
        ):
            assert model.training == False, f"The {name} is not in eval mode."
            for param in model.parameters():
                assert param.requires_grad == False, f"The {name} is not frozen."

    def _query_teacher(
        self, z_scaled, t, prompt_embeds, prompt_mask, guidance_scale=None
    ):
        """ This helper function takes care of classifier-free guidance
            The last argument (guidance_scale) is only effective when using variable
            guidance scale, i.e., self.teacher_guidance_scale is -1
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        if len(t.reshape(-1)) != 1 and self.use_teacher_cf_guidance:
            t = torch.cat([t] * 2)

        z_scaled_cat = (
            torch.cat([z_scaled] * 2) if self.use_teacher_cf_guidance else z_scaled
        )

        # Get noise prediction from the teacher diffusion model (velocity by default)
        noise_pred = self.teacher_unet(
            z_scaled_cat, t, prompt_embeds, encoder_attention_mask=prompt_mask
        ).sample.detach()

        if self.use_teacher_cf_guidance:

            if self.teacher_guidance_scale == -1:
                if not torch.is_tensor(guidance_scale):
                    guidance_scale = torch.tensor(guidance_scale)
                cur_guidance_scale = guidance_scale.to(noise_pred.device)
                cur_guidance_scale = cur_guidance_scale.reshape(-1, 1, 1, 1)
            else:
                cur_guidance_scale = self.teacher_guidance_scale

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = (1 - cur_guidance_scale) * noise_pred_uncond + \
                cur_guidance_scale * noise_pred_cond

        assert not noise_pred.isnan().any(), f"noise_pred is NaN at t={t}"
        return noise_pred

    def forward(self, z_0, prompt, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, prompt, inference_scheduler, **kwargs):
        raise NotImplementedError
