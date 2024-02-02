from copy import deepcopy
from typing import Any, Mapping
from collections import OrderedDict
from time import time

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

from diffusers.utils import randn_tensor
from diffusers import DDIMScheduler, HeunDiscreteScheduler
from models.audio_distilled_model import AudioDistilledModel
from tools.train_utils import do_ema_update
from tools.losses import MSELoss, MelLoss, CLAPLoss, MultiResolutionSTFTLoss

logger = get_logger(__name__, log_level="INFO")


class AudioLCM(AudioDistilledModel):

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
        loss_type='mse'
    ):
        super().__init__(
            text_encoder_name=text_encoder_name,
            scheduler_name=scheduler_name,
            unet_model_name=unet_model_name,
            unet_model_config_path=unet_model_config_path,
            snr_gamma=snr_gamma,
            freeze_text_encoder=freeze_text_encoder,
            use_lora=use_lora,
            ema_decay=ema_decay,
            teacher_guidance_scale=teacher_guidance_scale
        )

        assert unet_model_name is not None or unet_model_config_path is not None, \
            "Either UNet pretrain model name or a config file path is required"

        self.uncondition = uncondition  # If true, randomly drop 10% text condition
        self.use_edm = use_edm
        self.use_karras = use_karras
        self.target_ema_decay = target_ema_decay
        self.num_diffusion_steps = num_diffusion_steps

        self.lightweight = 'light' in unet_model_config_path
        logger.info(f"Use the lightweight model setting: {self.lightweight}")

        # Instantiate the target UNet, which only applies for stage-2
        self.student_target_unet = deepcopy(self.student_unet)
        self.student_target_unet.eval()
        self.student_target_unet.requires_grad_(False)

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        # Initialize noise scheduler. It is used for querying
        # the diffusion model during consistency distillation
        sched_class = HeunDiscreteScheduler if self.use_edm else DDIMScheduler
        self.noise_scheduler = sched_class.from_pretrained(
            self.scheduler_name, subfolder="scheduler"
        )

        if self.use_karras:
            if self.use_edm:
                logger.info("Using Karras noise schedule.")
                self.noise_scheduler.use_karras_sigmas = True
            else:
                ValueError("Karras noise schedule can only be used with Heun scheduler.")

        self.noise_scheduler.set_timesteps(self.num_diffusion_steps, device=self.device)
        # logger.info(f"Available timesteps: {self.noise_scheduler.timesteps.tolist()}")
        # logger.info("Noise scheduler prediction type: "
        #             f"{self.noise_scheduler.config.prediction_type}")

        self.vae = vae
        self.loss_type = loss_type

        self.losses = {
            'mse': MSELoss(reduction='instance'),
            'mel': MelLoss(vae=self.vae, reduction='instance'),
            'stft': MultiResolutionSTFTLoss(
                vae=self.vae, reduction='instance', fft_sizes=[1024, 2048, 512],
                hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240],
                window="hann_window", factor_sc=0.1, factor_mag=0.1, factor_mse=.8
            ),
            'clap': CLAPLoss(
                vae=self.vae, reduction='instance', mse_weight=1., clap_weight=.1
            )
        }
        logger.info(f"Using the {self.loss_type} loss.")
        self.loss = self.losses[self.loss_type]

    def load_state_dict_from_tango(
        self, tango_state_dict: Mapping[str, Any],
        stage1_state_dict: Mapping[str, Any] = None
    ):
        """ Load the teacher diffusion model from a pre-trained TANGO checkpoint;
            initialize the student model and its EMA copies with the teacher weights.
        """
        new_state_dict = OrderedDict()

        student_modules = ["student", "student_target", "student_ema"]
        for key, val in tango_state_dict.items():
            if 'unet' in key and '_unet' not in key:
                new_state_dict[f"teacher_{key}"] = val
                if stage1_state_dict is None:
                    for module in student_modules:
                        new_state_dict[f"{module}_{key}"] = val
            else:
                new_state_dict[key] = val

        if stage1_state_dict is not None:
            for key, val in stage1_state_dict.items():
                if 'student_ema' in key:
                    aft_key = key.split('student_ema_')[-1]
                    for module in student_modules:
                        new_state_dict[f"{module}_{aft_key}"] = val

        try:
            return_info = self.load_state_dict(new_state_dict, strict=True)
        except:
            logger.info(
                "Strict loading failed. The loaded state_dict may not match the target model. "
                "This is okay if 'Keys that are not loaded' is an empty list."
            )
            return_info = self.load_state_dict(new_state_dict, strict=False)
            missing_keys = [
                key for key in return_info.missing_keys if 'vae' not in key and 'loss.' not in key
            ]
            redundant_keys = [
                key for key in return_info.unexpected_keys if 'vae' not in key and 'loss.' not in key
            ]
            print(f"Keys that are not loaded: {missing_keys}")
            assert len(redundant_keys) == 0, \
                f"Redundant keys in state_dict: {return_info.unexpected_keys}"

        # Low-rank adaptation
        if self.use_lora:
            self.setup_lora()

        self.student_target_unet.requires_grad_(False)
        self.student_ema_unet.requires_grad_(False)
        self.teacher_unet.requires_grad_(False)
        return return_info

    def load_pretrained(self, state_dict: Mapping[str, Any], strict: bool = True):
        """ This function converts parameter names before loading the state_dict,
            so that we can use the model trained via older implementations.
        """
        new_state_dict = OrderedDict()

        for key, val in state_dict.items():
            if 'consistency_unet' in key:
                aft_key = key.split('consistency_unet')[-1]
                new_state_dict[f"student_unet{aft_key}"] = val
            elif 'consistency_ema_' in key:
                aft_key = key.split('consistency_ema_')[-1]
                new_state_dict[f"student_target_{aft_key}"] = val
                if f"student_ema_{aft_key}" not in new_state_dict.keys():
                    new_state_dict[f"student_ema_{aft_key}"] = val
            elif 'consistency_slow_ema_' in key:
                aft_key = key.split('consistency_slow_ema_')[-1]
                new_state_dict[f"student_ema_{aft_key}"] = val
            elif 'diffusion_unet' in key:
                aft_key = key.split('diffusion_unet')[-1]
                new_state_dict[f"teacher_unet{aft_key}"] = val
            elif "loss." in key and "vae." not in key:  # Do not load the VAE
                aft_key = key.split('loss.')[-1]
                new_state_dict[f"stft_loss.{aft_key}"] = val
            elif "vae." not in key:  # Do not load the VAE
                new_state_dict[key] = val

        try:
            return self.load_state_dict(new_state_dict, strict=True)
        except:
            logger.info(
                "Strict loading failed. The loaded state_dict may not match the target model. "
                "This is okay if 'Keys that are not loaded' is an empty list."
            )
            return_info = self.load_state_dict(new_state_dict, strict=False)
            missing_keys = [
                key for key in return_info.missing_keys if 'vae' not in key and 'loss.' not in key
            ]
            redundant_keys = [
                key for key in return_info.unexpected_keys if 'vae' not in key and 'loss.' not in key
            ]
            print(f"Keys that are not loaded: {missing_keys}")
            assert len(redundant_keys) == 0, \
                f"Redundant keys in state_dict: {return_info.unexpected_keys}"
            return return_info

    def train(self, mode: bool = True):
        super().train(mode)
        self.student_target_unet.eval()
        self.vae.eval()
        return self

    def eval(self):
        return self.train(mode=False)

    def compute_snr(self, timesteps, t_indices):
        if self.use_edm:
            return self.noise_scheduler.sigmas[t_indices] ** (-2)
        else:
            return super().compute_snr(timesteps)

    def update_ema(self):
        assert self.training, "EMA update should only be called during training"
        do_ema_update(
            source_model=self.student_unet,
            shadow_models=[self.student_target_unet, self.student_ema_unet],
            decay_consts=[self.target_ema_decay, self.ema_decay]
        )

    def check_eval_mode(self):
        super().check_eval_mode()
        # Check the student target UNet, which is only in stage-2
        for model, name in zip(
            [self.student_target_unet, self.vae], ['student_target_unet', 'vae']
        ):
            assert model.training == False, f"The {name} is not in eval mode."
            for param in model.parameters():
                assert param.requires_grad == False, f"The {name} is not frozen."

    def forward(self, z_0, gt_wav, prompt, validation_mode=False, run_teacher=True, **kwargs):
        """
        z_0:                Ground-truth latent variables.
        prompt:             Text prompt for the generation.
        validation_mode:    If 0 or False, operate in training mode and sample a random
                                timestep. If >0, operate in validation model, and then it
                                specifies the index of the discrite time step.
        run_teacher:        If True, run the teacher all the way to t=0 for validation
                                loss calculation. Otherwise, only query the teacher once.
        """

        def get_loss(model_pred, target, gt_wav, prompt, timesteps, t_indices):
            if self.snr_gamma is None:
                return self.loss(model_pred, target, gt_wav, prompt).mean()

            else:
                if not torch.is_tensor(timesteps):
                    timesteps = torch.tensor(timesteps)
                assert len(timesteps.shape) < 2
                timesteps = timesteps.reshape(-1)

                # Compute loss weights as per Section 3.4 of arxiv.org/abs/2303.09556
                snr = self.compute_snr(timesteps, t_indices).reshape(-1)
                mse_loss_weights = torch.clamp(snr, max=self.snr_gamma)

                # Compute weighted loss
                instance_loss = self.loss(model_pred, target, gt_wav, prompt)
                return (instance_loss * mse_loss_weights.to(instance_loss.device)).mean()

        def get_random_timestep(batch_size, validation_mode):
            # Get available time steps
            # The time steps spread between 0 and training time steps (1000).
            device = self.text_encoder.device
            avail_timesteps = self.noise_scheduler.timesteps.to(device)
            order = 2 if self.use_edm else 1

            if validation_mode != 0:  # Validation mode
                # Run specified number of steps
                time_ind = len(avail_timesteps) - 1 - int(validation_mode * order)
                assert time_ind >= 0
                time_inds = time_ind * torch.ones(
                    (batch_size,), dtype=torch.int32, device=device
                )
            else:  # Training mode
                # Sample a random timestep for each instance
                time_inds = torch.randint(
                    0, (len(avail_timesteps) - 1) // order, (batch_size,), device=device
                ) * order

            t_nplus1 = avail_timesteps[time_inds]
            t_n = avail_timesteps[time_inds + order]
            return t_nplus1, t_n, time_inds, time_inds + order

        # Check if the relevant models are in eval mode and frozen
        self.check_eval_mode()
        assert validation_mode >= 0

        # Encode text; this is unchanged compared with TANGO
        prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = \
            self.get_prompt_embeds(
                prompt, self.use_teacher_cf_guidance, num_samples_per_prompt=1
            )

        # Randomly mask 10% of the encoder hidden states
        if self.uncondition:
            raise NotImplementedError

        # Sample a random available time step
        t_nplus1, t_n, t_ind_nplus1, t_ind_n = get_random_timestep(
            z_0.shape[0], validation_mode
        )

        # Noisy latents
        gaussian_noise = torch.randn_like(z_0)
        z_noisy = self.noise_scheduler.add_noise(z_0, gaussian_noise, t_nplus1)
        z_gaussian = gaussian_noise * self.noise_scheduler.init_noise_sigma

        # Resample the final time step
        last_step = self.noise_scheduler.timesteps.max()
        last_mask = (t_nplus1 == last_step).reshape(-1, 1, 1, 1)
        z_nplus1 = torch.where(last_mask, z_gaussian.to(z_0.device), z_noisy)
        z_nplus1_scaled = self.noise_scheduler.scale_model_input(z_nplus1, t_nplus1)

        if self.use_edm:
            assert self.noise_scheduler.state_in_first_order

        if self.teacher_guidance_scale == -1:  # Random guidance scale
            guidance_scale = torch.rand(z_0.shape[0]) * self.max_rand_guidance_scale
            guidance_scale = guidance_scale.to(z_0.device) 
        else:
            guidance_scale = None

        # Query the diffusion teacher model
        with torch.no_grad():
            noise_pred_nplus1 = self._query_teacher(
                z_nplus1_scaled, t_nplus1, prompt_embeds_cf, prompt_mask_cf, guidance_scale
            )
            # Recover estimation of z_n from the noise prediction
            zhat_n = self.noise_scheduler.step(
                noise_pred_nplus1, t_nplus1, z_nplus1
            ).prev_sample
            zhat_n_scaled = self.noise_scheduler.scale_model_input(zhat_n, t_n)
            assert not zhat_n_scaled.isnan().any(), f"zhat_n is NaN at t={t_nplus1}"

            if self.use_edm:  # EDM requires two teacher queries
                noise_pred_n = self._query_teacher(
                    zhat_n_scaled, t_n, prompt_embeds_cf, prompt_mask_cf, guidance_scale
                )
                # Step scheduler again to perform Heun update
                zhat_n = self.noise_scheduler.step(noise_pred_n, t_n, zhat_n).prev_sample
                zhat_n_scaled = self.noise_scheduler.scale_model_input(zhat_n, t_n)
                assert not zhat_n_scaled.isnan().any(), f"zhat_n is NaN at t={t_nplus1}"
                assert self.noise_scheduler.state_in_first_order

        # Query the diffusion model to obtain the estimation of z_n
        if validation_mode != 0:
            with torch.no_grad():
                zhat_0_from_nplus1 = self.student_target_unet(
                    z_nplus1_scaled, t_nplus1, guidance=guidance_scale,
                    encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
                ).sample

                zhat_0_from_n = self.student_target_unet(
                    zhat_n_scaled, t_n, guidance=guidance_scale,
                    encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
                ).sample

                if run_teacher:
                    device = self.text_encoder.device
                    avail_timesteps = self.noise_scheduler.timesteps.to(device)

                    for t in avail_timesteps[t_ind_n[0]:]:
                        # Get noise prediction from the diffusion model
                        zhat_n_scaled_tea = self.noise_scheduler.scale_model_input(zhat_n, t)
                        noise_pred_n = self._query_teacher(
                            zhat_n_scaled_tea, t, prompt_embeds_cf, prompt_mask_cf, guidance_scale
                        )
                        # Step scheduler
                        zhat_n = self.noise_scheduler.step(noise_pred_n, t, zhat_n)
                        zhat_n = zhat_n.prev_sample
                        assert not zhat_n.isnan().any()

                    # logger.info(f"loss w/ gt: {F.mse_loss(zhat_0_from_nplus1, z_0).item()}")
                    # logger.info(
                    #     f"loss w/ teacher: {F.mse_loss(zhat_0_from_nplus1, zhat_n).item()}"
                    # )
                    # loss_cons_ = get_loss(
                    #     zhat_0_from_nplus1, zhat_0_from_n, 
                    #     avail_timesteps[t_ind_nplus1[0]], t_ind_nplus1[0]
                    # ).item()
                    # logger.info(f"consistency loss: {loss_cons_}")
                    # logger.info(f"teacher loss: {F.mse_loss(zhat_0_from_n, zhat_n).item()}")

                    if self.use_edm:
                        self.noise_scheduler.prev_derivative = None
                        self.noise_scheduler.dt = None
                        self.noise_scheduler.sample = None

            t_nplus1 = avail_timesteps[t_ind_nplus1[0]]
            loss_w_gt = F.mse_loss(zhat_0_from_nplus1, z_0)  # w.r.t. ground truth
            loss_w_teacher = F.mse_loss(zhat_0_from_nplus1, zhat_n)  # w.r.t. teacher model
            loss_consis = get_loss(
                zhat_0_from_nplus1, zhat_0_from_n, gt_wav, prompt, t_nplus1, t_ind_nplus1[0]
            )
            loss_teacher = F.mse_loss(zhat_n, z_0)  # teacher loss

            return loss_w_gt, loss_w_teacher, loss_consis, loss_teacher

        else:  # Training mode

            with torch.no_grad():
                # Feed both z_n and z_{n+1} into the consistency model and minimize the loss
                zhat_0_from_n = self.student_target_unet(
                    zhat_n_scaled, t_n, guidance=guidance_scale,
                    encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
                ).sample.detach()
                # If t_n is 0, use ground truth latent as the target
                zhat_0_from_n = torch.where(
                    (t_n == 0).reshape(-1, 1, 1, 1), z_0, zhat_0_from_n
                )

            zhat_0_from_nplus1 = self.student_unet(
                z_nplus1_scaled, t_nplus1, guidance=guidance_scale,
                encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
            ).sample

            return get_loss(
                zhat_0_from_nplus1, zhat_0_from_n, gt_wav, prompt, t_nplus1, t_ind_nplus1
            )

    @torch.no_grad()
    def inference(
        self, prompt, inference_scheduler, guidance_scale_input=3, guidance_scale_post=1,
        num_steps=20, use_edm=False, num_samples=1, use_ema=True, 
        query_teacher=False, num_teacher_steps=18, return_all=False
    ):
        def calc_zhat_0(
            z_n, t, prompt_embeds, prompt_mask, guidance_scale_input, guidance_scale_post
        ):
            use_cf_guidance = guidance_scale_post > 1.

            # expand the latents if we are doing classifier free guidance
            z_n_input = torch.cat([z_n] * 2) if use_cf_guidance else z_n
            # Scale model input as required for some schedules.
            z_n_input = inference_scheduler.scale_model_input(z_n_input, t)

            # Get zhat_0 from the model
            unet = self.student_ema_unet if use_ema else self.student_target_unet
            zhat_0 = unet(
                z_n_input, t, guidance=guidance_scale_input,
                encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
            ).sample

            # perform guidance
            if use_cf_guidance:
                zhat_0_uncond, zhat_0_cond = zhat_0.chunk(2)
                zhat_0 = (1 - guidance_scale_post) * zhat_0_uncond + \
                    guidance_scale_post * zhat_0_cond
            return zhat_0

        self.check_eval_mode()
        device = self.text_encoder.device
        batch_size = len(prompt) * num_samples
        use_cf_guidance = guidance_scale_post > 1.

        # Get prompt embeddings
        t_start_embed = time()
        prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = \
            self.encode_text_classifier_free(prompt, num_samples)
        encoder_states_stu, encoder_att_mask_stu = \
            (prompt_embeds_cf, prompt_mask_cf) if use_cf_guidance \
                else (prompt_embeds, prompt_mask)
        encoder_states_tea, encoder_att_mask_tea = \
            (prompt_embeds_cf, prompt_mask_cf) if self.use_teacher_cf_guidance \
                else (prompt_embeds, prompt_mask)

        # Prepare noise
        num_channels_latents = self.student_target_unet.config.in_channels
        latent_shape = (batch_size, num_channels_latents, 256, 16)
        noise = randn_tensor(
            latent_shape, generator=None, device=device, dtype=prompt_embeds.dtype
        )
        time_embed = time() - t_start_embed

        # Query the inference scheduler to obtain the time steps.
        # The time steps spread between 0 and training time steps
        t_start_stu = time()
        inference_scheduler.set_timesteps(18, device=device)
        z_N_stu = noise * inference_scheduler.init_noise_sigma

        # Query the consistency model
        zhat_0_stu = calc_zhat_0(
            z_N_stu, inference_scheduler.timesteps[0], encoder_states_stu,
            encoder_att_mask_stu, guidance_scale_input, guidance_scale_post
        )

        # Iteratively query the consistency model if requested
        inference_scheduler.set_timesteps(num_steps, device=device)
        order = 2 if self.use_edm else 1

        for t in inference_scheduler.timesteps[1::order]:
            zhat_n_stu = inference_scheduler.add_noise(
                zhat_0_stu, torch.randn_like(zhat_0_stu), t
            )
            # Calculate new zhat_0
            zhat_0_stu = calc_zhat_0(
                zhat_n_stu, t, encoder_states_stu, encoder_att_mask_stu,
                guidance_scale_input, guidance_scale_post
            )
        time_stu = time() - t_start_stu
        if return_all:
            print("Distilled model generation completed!")

        # Query the teacher model as well if requested by user
        if query_teacher:
            t_start_tea = time()
            inference_scheduler.set_timesteps(num_teacher_steps, device=device)
            zhat_n_tea = noise * inference_scheduler.init_noise_sigma

            for t in inference_scheduler.timesteps:
                zhat_n_input = inference_scheduler.scale_model_input(zhat_n_tea, t)
                noise_pred = self._query_teacher(
                    zhat_n_input, t, encoder_states_tea, encoder_att_mask_tea,
                    guidance_scale_input
                )
                zhat_n_tea = inference_scheduler.step(noise_pred, t, zhat_n_tea).prev_sample

            # Reset solver
            if self.use_edm:
                inference_scheduler.prev_derivative = None
                inference_scheduler.dt = None
                inference_scheduler.sample = None

            # loss_w_teacher = ((zhat_0_stu - zhat_n_tea) ** 2).mean().item()
            # logger.info(f"loss w.r.t. teacher: {loss_w_teacher}")
            time_tea = time() - t_start_tea
            if return_all:
                print("Diffusion model generation completed!")

        else:
            zhat_n_tea, time_tea = None, None

        if return_all:
            # Return student generation, teacher generation, student time, teacher time
            if time_tea is not None:
                time_tea += time_embed
            return zhat_0_stu, zhat_n_tea, time_stu + time_embed, time_tea
        else:
            # Return student generation
            return zhat_0_stu
