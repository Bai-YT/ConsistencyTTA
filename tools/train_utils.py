from tqdm.auto import tqdm
import os
import json
import math
import numpy as np
from collections import OrderedDict

import torch
from transformers import get_scheduler
from accelerate.logging import get_logger
logger = get_logger(__name__)
import wandb

import diffusers
from tools import torch_tools

TARGET_LENGTH = 1024


def get_optimizer_and_scheduler(args, model, train_dataloader, accelerator):

    # Set optimization variables
    if args.use_lora:
        assert not args.finetune_vae, \
            "Fine-tuning VAE with LoRA has not been implemented."
        lora_layers = diffusers.loaders.AttnProcsLayers(
            model.student_unet.attn_processors
        )
        optimizer_parameters = lora_layers.parameters()
        logger.info("Optimizing LoRA parameters.")
    elif args.finetune_vae:
        optimizer_parameters = (
            list(model.student_unet.parameters()) +
            list(model.vae.decoder.parameters()) +
            list(model.vae.post_quant_conv.parameters())
        )
        logger.info("Optimizing UNet and VAE decoder parameters.")
    else:
        optimizer_parameters = model.student_unet.parameters()
        logger.info("Optimizing UNet parameters.")

    num_unet_trainable_parameters = sum(
        p.numel() for p in model.student_unet.parameters() if p.requires_grad
    )
    num_vae_trainable_parameters = sum(
        p.numel() for p in model.vae.parameters() if p.requires_grad
    )
    num_total_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_other_trainable_parameters = num_total_trainable_parameters \
        - num_unet_trainable_parameters - num_vae_trainable_parameters

    logger.info(f"Num trainable U-Net parameters: {num_unet_trainable_parameters}.")
    logger.info(f"Num trainable VAE parameters: {num_vae_trainable_parameters}.")
    logger.info(f"Num trainable other parameters: {num_other_trainable_parameters}.")
    logger.info(f"Num trainable total parameters: {num_total_trainable_parameters}.")

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    optimizer.zero_grad()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
    )

    return optimizer, lr_scheduler, overrode_max_train_steps


def eval_model(
    model, vae, stft, stage, eval_dataloader, accelerator, num_diffusion_steps
):
    model.eval()
    model.uncondition = False

    num_data_to_eval = len(eval_dataloader.dataset) if stage == 1 else 100
    num_losses = 4 if stage >= 2 else 1
    total_val_losses = [[] for _ in range(num_losses)]

    # eval_steps = list(
    #     (2 ** (np.arange(0, np.log2(num_diffusion_steps) - 1))).astype(int)
    # ) + [num_diffusion_steps - 1]
    eval_steps = [num_diffusion_steps - 1]

    eval_progress_bar = tqdm(
        range(num_data_to_eval * len(eval_steps)),
        disable=not accelerator.is_local_main_process
    )

    for validation_mode in eval_steps:
        total_val_loss = [0 for _ in range(num_losses)]
        num_tested = 0
    
        for cntr, (captions, gt_waves, _) in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():

                unwrapped_vae = accelerator.unwrap_model(vae)
                mel, _ = torch_tools.wav_to_fbank(gt_waves, TARGET_LENGTH, stft)
                mel = mel.unsqueeze(1).to(model.device)
                true_latent = unwrapped_vae.get_first_stage_encoding(
                    unwrapped_vae.encode_first_stage(mel)
                )

                val_loss = model(
                    true_latent, gt_waves, captions,validation_mode=validation_mode
                )
                eval_progress_bar.update(len(captions) * torch.cuda.device_count())

                if not isinstance(val_loss, tuple) and not isinstance(val_loss, list):
                    val_loss = [val_loss]
                for i in range(num_losses):
                    total_val_loss[i] += val_loss[i].detach().float().item()

                num_tested += len(captions) * torch.cuda.device_count()
                if num_tested >= num_data_to_eval:
                    break

        accelerator.print()
        for i in range(num_losses):
            total_val_losses[i] += [total_val_loss[i] / cntr]
            logger.info(
                f"{validation_mode} steps loss {i + 1}: {total_val_losses[i][-1]}"
            )

    return [np.array(tvl).mean() for tvl in total_val_losses]


def train_one_epoch(
    model, vae, stft, train_dataloader, accelerator, args, optimizer,
    lr_scheduler, checkpointing_steps, completed_steps, progress_bar
):
    model.train()
    model.uncondition = args.uncondition
    total_loss = 0

    for captions, gt_waves, _ in train_dataloader:

        with accelerator.accumulate(model):

            # Calculate latent embeddings
            with torch.no_grad():
                unwrapped_vae = accelerator.unwrap_model(vae)
                mel, _ = torch_tools.wav_to_fbank(gt_waves, TARGET_LENGTH, stft)
                mel = mel.unsqueeze(1).to(model.device)

                true_x0 = unwrapped_vae.get_first_stage_encoding(
                    unwrapped_vae.encode_first_stage(mel)
                )

            # Calculate gradient and perform optimization step
            loss = model(true_x0, gt_waves, captions, validation_mode=False)
            accelerator.backward(loss)
            if not torch.isnan(loss):
                total_loss += loss.detach().float().item()
                optimizer.step()
                lr_scheduler.step()
            else:
                logger.info("NaN loss encountered.")
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step
        # behind the scenes
        if accelerator.sync_gradients:
            # Update the EMA consistency model
            with torch.no_grad():
                if torch.cuda.device_count() == 1:
                    model.update_ema()
                else:
                    model.module.update_ema()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({ 
                'lr': optimizer.param_groups[0]['lr'], 'train_loss': loss.item()
            })
            completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps }"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= args.max_train_steps:
            break

    return total_loss, completed_steps


def log_results(
    accelerator, logger, epoch, completed_steps, lr, train_loss,
    val_loss, best_eval_loss, output_dir, with_tracking
):
    save_checkpoint = False

    if accelerator.is_main_process:    
        result = {}
        result["epoch"] = epoch,
        result["step"] = completed_steps
        result["lr"] = lr

        if len(val_loss) == 4:  # Stage-2 distillation
            result["loss_wrt_gt"] = round(val_loss[0], 6)
            result["loss_wrt_teacher"] = round(val_loss[1], 6)
            result["consistency_loss"] = round(val_loss[2], 6)
            result["teacher_loss"] = round(val_loss[3], 6)
            result_string = (f"Epoch: {epoch}, "
                             f"Val loss wrt teacher: {val_loss[1]:.4f}, ")
            loss_to_track = result["loss_wrt_teacher"]

        else:  # Stage-1 distillation
            result["validation_loss"] = round(val_loss[0], 6)
            result_string = f"Epoch: {epoch}, Val loss: {val_loss[0]:.4f}, "
            loss_to_track = result["validation_loss"]

        if train_loss is not None:
            result["train_loss"] = round(train_loss, 6)

        wandb.log(result)

        if train_loss is not None:
            result_string += f"Training loss: {train_loss:.4f}\n"
        logger.info(result_string)

        with open(f"{output_dir}/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")

        logger.info(result)

        if loss_to_track < best_eval_loss:
            best_eval_loss = loss_to_track
            save_checkpoint = True

    if with_tracking:
        accelerator.log(result, step=completed_steps)

    return save_checkpoint, best_eval_loss


def do_ema_update(source_model, shadow_models, decay_consts):
    """Performs the exponential model average (EMA) update.

    Args:
        source_model:   The source model.
        shadow_models:  A list of shadow models to be updated.
        decay_consts:   A list of EMA decay constants
                        corresponding to the shadow models.
    """
    assert len(shadow_models) == len(decay_consts)
    model_params = OrderedDict(source_model.named_parameters())
    model_buffers = OrderedDict(source_model.named_buffers())

    for shadow_model, ema_decay in zip(shadow_models, decay_consts):
        shadow_params = OrderedDict(shadow_model.named_parameters())
        shadow_buffers = OrderedDict(shadow_model.named_buffers())

        # check if both models contain the same set of keys
        assert ema_decay <= 1 and ema_decay >= 0
        assert model_params.keys() == shadow_params.keys()
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, param in model_params.items():  # Copy parameters
            shadow_params[name].add_(
                (1. - ema_decay) * (param - shadow_params[name])
            )
        for name, buffer in model_buffers.items():  # Copy buffer
            shadow_buffers[name].copy_(buffer)
