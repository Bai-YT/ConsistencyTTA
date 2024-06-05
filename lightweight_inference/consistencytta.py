import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, T5EncoderModel

from diffusers.utils.torch_utils import randn_tensor
from diffusers import UNet2DConditionGuidedModel, HeunDiscreteScheduler
from audioldm.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config


class ConsistencyTTA(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize the consistency U-Net
        unet_model_config_path='tango_diffusion_light.json'
        unet_config = UNet2DConditionGuidedModel.load_config(unet_model_config_path)
        self.unet = UNet2DConditionGuidedModel.from_config(unet_config, subfolder="unet")

        unet_weight_path = "consistencytta_clapft_ckpt/unet_state_dict.pt"
        unet_weight_sd = torch.load(unet_weight_path, map_location='cpu')
        self.unet.load_state_dict(unet_weight_sd)

        # Initialize FLAN-T5 tokenizer and text encoder
        text_encoder_name = 'google/flan-t5-large'
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
        self.text_encoder.eval(); self.text_encoder.requires_grad_(False)

        # Initialize the VAE
        raw_vae_path = "consistencytta_clapft_ckpt/vae_state_dict.pt"
        raw_vae_sd = torch.load(raw_vae_path, map_location="cpu")
        vae_state_dict, scale_factor = raw_vae_sd["state_dict"], raw_vae_sd["scale_factor"]

        config = default_audioldm_config('audioldm-s-full')
        vae_config = config["model"]["params"]["first_stage_config"]["params"]
        vae_config["scale_factor"] = scale_factor

        self.vae = AutoencoderKL(**vae_config)
        self.vae.load_state_dict(vae_state_dict)
        self.vae.eval(); self.vae.requires_grad_(False)

        # Initialize the STFT
        self.fn_STFT = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],  # default 1024
            config["preprocessing"]["stft"]["hop_length"],  # default 160
            config["preprocessing"]["stft"]["win_length"],  # default 1024
            config["preprocessing"]["mel"]["n_mel_channels"],  # default 64
            config["preprocessing"]["audio"]["sampling_rate"],  # default 16000
            config["preprocessing"]["mel"]["mel_fmin"],  # default 0
            config["preprocessing"]["mel"]["mel_fmax"],  # default 8000
        )
        self.fn_STFT.eval(); self.fn_STFT.requires_grad_(False)

        self.scheduler = HeunDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path='stabilityai/stable-diffusion-2-1', subfolder="scheduler"
        )


    def train(self, mode: bool = True):
        self.unet.train(mode)
        for model in [self.text_encoder, self.vae, self.fn_STFT]:
            model.eval()
        return self


    def eval(self):
        return self.train(mode=False)


    def check_eval_mode(self):
        for model, name in zip(
            [self.text_encoder, self.vae, self.fn_STFT, self.unet],
            ['text_encoder', 'vae', 'fn_STFT', 'unet']
        ):
            assert model.training == False, f"The {name} is not in eval mode."
            for param in model.parameters():
                assert param.requires_grad == False, f"The {name} is not frozen."


    @torch.no_grad()
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

        prompt_embeds = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        bool_prompt_mask = (attention_mask == 1).to(device)  # Convert to boolean
        return prompt_embeds, bool_prompt_mask


    @torch.no_grad()
    def encode_text_classifier_free(self, prompt: str, num_samples_per_prompt: int):
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

        """ For classifier-free guidance, we need to do two forward passes.
            We concatenate the unconditional and text embeddings into a single batch 
        """
        prompt_embeds = torch.cat([negative_prompt_embeds, cond_prompt_embeds])
        prompt_mask = torch.cat([uncond_prompt_mask, cond_prompt_mask])

        return prompt_embeds, prompt_mask, cond_prompt_embeds, cond_prompt_mask


    def forward(
        self, prompt: str, cfg_scale_input: float = 3., cfg_scale_post: float = 1.,
        num_steps: int = 1, num_samples: int = 1, sr: int = 16000
    ):
        self.check_eval_mode()
        device = self.text_encoder.device
        use_cf_guidance = cfg_scale_post > 1.

        # Get prompt embeddings
        prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = \
            self.encode_text_classifier_free(prompt, num_samples)
        encoder_states, encoder_att_mask = \
            (prompt_embeds_cf, prompt_mask_cf) if use_cf_guidance \
                else (prompt_embeds, prompt_mask)

        # Prepare noise
        num_channels_latents = self.unet.config.in_channels
        latent_shape = (len(prompt) * num_samples, num_channels_latents, 256, 16)
        noise = randn_tensor(
            latent_shape, generator=None, device=device, dtype=prompt_embeds.dtype
        )

        # Query the inference scheduler to obtain the time steps.
        # The time steps spread between 0 and training time steps
        self.scheduler.set_timesteps(18, device=device)  # Set this to training steps first
        z_N = noise * self.scheduler.init_noise_sigma

        def calc_zhat_0(z_n: Tensor, t: int):
            """ Query the consistency model to get zhat_0, which is the denoised embedding.
            Args:
                z_n (Tensor):   The noisy embedding.
                t (int):        The time step.
            Returns:
                Tensor:         The denoised embedding.
            """
            # expand the latents if we are doing classifier free guidance
            z_n_input = torch.cat([z_n] * 2) if use_cf_guidance else z_n
            # Scale model input as required for some schedules.
            z_n_input = self.scheduler.scale_model_input(z_n_input, t)

            # Get zhat_0 from the model
            zhat_0 = self.unet(
                z_n_input, t, guidance=cfg_scale_input,
                encoder_hidden_states=encoder_states, encoder_attention_mask=encoder_att_mask
            ).sample

            # Perform external classifier-free guidance
            if use_cf_guidance:
                zhat_0_uncond, zhat_0_cond = zhat_0.chunk(2)
                zhat_0 = (1 - cfg_scale_post) * zhat_0_uncond + cfg_scale_post * zhat_0_cond

            return zhat_0

        # Query the consistency model
        zhat_0 = calc_zhat_0(z_N, self.scheduler.timesteps[0])

        # Iteratively query the consistency model if requested
        self.scheduler.set_timesteps(num_steps, device=device)

        for t in self.scheduler.timesteps[1::2]:  # 2 is the order of the scheduler
            zhat_n = self.scheduler.add_noise(zhat_0, torch.randn_like(zhat_0), t)
            # Calculate new zhat_0
            zhat_0 = calc_zhat_0(zhat_n, t)

        mel = self.vae.decode_first_stage(zhat_0.float())
        return self.vae.decode_to_waveform(mel)[:, :int(sr * 9.5)]  # Truncate to 9.6 seconds
