from time import time
import argparse
import logging
import os
import json
from tqdm.auto import tqdm
import math
import numpy as np
import wandb

import torch
import datasets
import transformers
from transformers import SchedulerType

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
logger = get_logger(__name__)

import diffusers
from tools.t2a_dataset import get_dataloaders
from tools.build_pretrained import build_pretrained_models
from tools.train_utils import \
    train_one_epoch, eval_model, log_results, get_optimizer_and_scheduler
from tools.torch_tools import seed_all
from models import AudioGDM, AudioLCM, AudioLCM_FTVAE

TARGET_LENGTH = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a diffusion model for text to audio generation task."
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], default=2,
        help=("Specifies the stage of the disillation. Must be 1 or 2. "
              "Stage 2 corresponds to consistency distillation")
    )
    parser.add_argument(
        "--train_file", type=str, default="data/train_audiocaps.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--use_bf16", action="store_true", default=False,
        help="Use bf16 mixed precision training.",
    )
    parser.add_argument(
        "--use_lora", action="store_true", default=False, help="Use low-rank adaptation."
    )
    parser.add_argument(
        "--validation_file", type=str, default="data/valid_audiocaps.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="A csv or a json file containing the test data for generation."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1",
        help="Scheduler identifier.",
    )
    parser.add_argument(
        "--unet_model_name", type=str, default=None,
        help="UNet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default=None,
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--tango_model", type=str, default=None,
        help="Tango model identifier from huggingface: declare-lab/tango"
    )
    parser.add_argument(
        "--stage1_model", type=str, default=None,
        help="Path to the stage-one pretrained model. This effective only when stage is 2."
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
        help=("SNR weighting gamma to be used if rebalancing the loss. "
              "Recommended value is 5.0. Default to None. "
              "More details here: https://arxiv.org/abs/2303.09556.")
    )
    parser.add_argument(
        "--loss_type", type=str, default='mse', choices=['mse', 'mel', 'stft', 'clap'],
        help=("Loss type. Must be one of ['mse', 'mel', 'stft', 'clap']. "
              "This effective only when stage is 2.")
    )
    parser.add_argument(
        "--finetune_vae", action="store_true", default=False,
        help="Unfreeze the VAE parameters. Default is False."
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts."
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths."
    )
    parser.add_argument(
        "--augment", action="store_true", default=False, help="Augment training data."
    )
    parser.add_argument(
        "--uncondition", action="store_true", default=False,
        help="10% uncondition for training. Only applies to consistency distillation."
    )
    parser.add_argument(
        "--use_edm", action="store_true", default=False,
        help="Use the Heun solver proposed in EDM. Only applies to consistency distillation."
    )
    parser.add_argument(
        "--use_karras", action="store_true", default=False,
        help="Use the noise schedule proposed in EDM. Only effective when use_edm is True."
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="Add prefix in text prompts."
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=2,
        help="Batch size (per device) for the validation dataloader."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help=("Total number of training steps to perform. "
              "If provided, overrides num_train_epochs.")
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help=("Number of updates steps to accumulate before "
              "performing a backward/update pass.")
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", 
                 "polynomial", "constant", "constant_with_warmup"]
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer."
    )
    parser.add_argument(
        "--target_ema_decay", type=float, default=.95,
        help="Target network (for consistency ditillation) EMA decay rate. Default is 0.95."
    )
    parser.add_argument(
        "--ema_decay", type=float, default=.999,
        help="Exponential Model Average decay rate. Default is 0.999."
    )
    parser.add_argument(
        "--num_diffusion_steps", type=int, default=18,
        help=("Number of diffusion steps for the teacher model. "
              "Only applies to consistency distillation.")
    )
    parser.add_argument(
        "--teacher_guidance_scale", type=int, default=1,
        help=("The scale of classifier-free guidance used for the teacher model. "
              "If -1, then use random guidance scale drawn from Unif(0, 6).")
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help=("Whether the various states should be saved at the end of every "
              "'epoch' or 'best' whenever validation loss decreases.")
    )
    parser.add_argument(
        "--save_every", type=int, default=5,
        help="Save model after every how many epochs when checkpointing_steps is `best`."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--starting_epoch", type=int, default=0,
        help="The starting epoch (useful when resuming from checkpoint)",
    )
    parser.add_argument(
        "--eval_first", action="store_true",
        help="Whether to perform evaluation first before start training.",
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=("The integration to report the results and logs to. Supported "
              "platforms are `'tensorboard'`, `'wandb'`, `'comet_ml'` and `'clearml'`."
              "Use `'all'` (default) to report to all integrations."
              "Only applicable when `--with_tracking` is passed."),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], \
                "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], \
                "`validation_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        mixed_precision='bf16' if args.use_bf16 else 'no',
        **accelerator_log_kwargs
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        seed_all(args.seed)
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = f"saved/{int(time())}"

            if not os.path.exists("saved"):
                os.makedirs("saved")
            os.makedirs(args.output_dir, exist_ok=True)

        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs(f"{args.output_dir}/outputs", exist_ok=True)
        with open(f"{args.output_dir}/summary.jsonl", "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(project=f"Text to Audio Stage-{args.stage} Distillation")

    accelerator.wait_for_everyone()

    # Initialize models
    pretrained_model_name = "audioldm-s-full"
    vae, stft = build_pretrained_models(pretrained_model_name)
    # Freeze VAE and STFT
    vae.eval(); vae.requires_grad_(False)
    stft.eval(); stft.requires_grad_(False)

    # Create teacher-student model
    if args.stage == 1:
        model_class = AudioGDM
    elif args.finetune_vae:
        model_class = AudioLCM_FTVAE
    else:
        model_class = AudioLCM

    model = model_class(
        text_encoder_name=args.text_encoder_name,
        scheduler_name=args.scheduler_name,
        unet_model_name=args.unet_model_name,
        unet_model_config_path=args.unet_model_config,
        snr_gamma=args.snr_gamma,
        freeze_text_encoder=args.freeze_text_encoder,
        uncondition=args.uncondition,  # only effective for stage-2
        use_edm=args.use_edm,  # only effective for stage-2
        use_karras=args.use_karras,  # only effective for stage-2
        use_lora=args.use_lora,
        target_ema_decay=args.target_ema_decay,
        ema_decay=args.ema_decay,
        num_diffusion_steps=args.num_diffusion_steps,
        teacher_guidance_scale=args.teacher_guidance_scale,
        loss_type=args.loss_type,
        vae=vae
    )

    # Load pretrained TANGO checkpoint
    if args.tango_model is not None:
        # Important to load to CPU here so that VRAM is not occupied
        model.load_state_dict_from_tango(
            tango_state_dict=torch.load(args.tango_model, map_location='cpu'),
            stage1_state_dict=torch.load(args.stage1_model, map_location='cpu')
            if args.stage1_model is not None else None
        )
        logger.info(f"Loaded TANGO checkpoint from: {args.tango_model}")
        if args.stage == 2 and args.stage1_model is not None:
            logger.info(f"Loaded stage-1 checkpoint from: {args.stage1_model}")
    else:
        raise NotImplementedError

    # Freeze text encoder
    assert args.freeze_text_encoder, "Text encoder funetuning has not been implemented."
    assert args.unet_model_config, "unet_model_config must be specified."
    # Freeze text encoder
    model.text_encoder.eval()
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    logger.info("Text encoder is frozen.")

    # Dataloaders
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(args, accelerator)

    # Optimizer and LR scheduler
    optimizer, lr_scheduler, overrode_max_train_steps = get_optimizer_and_scheduler(
        args, model, train_dataloader, accelerator
    )

    # Prepare everything with accelerator
    vae, stft, model = vae.cuda(), stft.cuda(), model.cuda()
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, stft, model, optimizer, lr_scheduler
    )
    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    # We need to recalculate our total training steps as the size of
    # the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info(f"***** Running stage-{args.stage} training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = "
                f"{total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint, map_location='cpu')
            logger.info(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = args.starting_epoch
    completed_steps = starting_epoch * num_update_steps_per_epoch
    progress_bar.update(completed_steps - progress_bar.n)

    # Keep track of the best loss so far
    best_eval_loss = np.inf

    # Eval
    if args.eval_first:
        total_val_loss = eval_model(
            model=model, vae=vae, stft=stft, stage=args.stage,
            eval_dataloader=eval_dataloader, accelerator=accelerator,
            num_diffusion_steps=args.num_diffusion_steps
        )
        # Log results
        save_checkpoint, best_eval_loss = log_results(
            accelerator=accelerator,
            logger=logger,
            epoch=0,
            completed_steps=completed_steps,
            lr=optimizer.param_groups[0]['lr'],
            train_loss=None,
            val_loss=total_val_loss,
            best_eval_loss=best_eval_loss,
            output_dir=args.output_dir,
            with_tracking=args.with_tracking
        )

    for epoch in range(starting_epoch, args.num_train_epochs):
        # Train one epoch
        total_loss, completed_steps = train_one_epoch(
            model=model, vae=vae, stft=stft, train_dataloader=train_dataloader,
            accelerator=accelerator, args=args, optimizer=optimizer,
            lr_scheduler=lr_scheduler, checkpointing_steps=checkpointing_steps,
            completed_steps=completed_steps, progress_bar=progress_bar
        )    
        # Eval
        total_val_loss = eval_model(
            model=model, vae=vae, stft=stft, stage=args.stage,
            eval_dataloader=eval_dataloader, accelerator=accelerator,
            num_diffusion_steps=args.num_diffusion_steps
        )
        accelerator.wait_for_everyone()

        # Log results
        save_checkpoint, best_eval_loss = log_results(
            accelerator=accelerator,
            logger=logger,
            epoch=epoch + 1,
            completed_steps=completed_steps,
            lr=optimizer.param_groups[0]['lr'],
            train_loss=total_loss / len(train_dataloader),
            val_loss=total_val_loss,
            best_eval_loss=best_eval_loss,
            output_dir=args.output_dir,
            with_tracking=args.with_tracking
        )

        model_saved = False
        while not model_saved:  # Make sure the model is successfully saved
            try:
                if accelerator.is_main_process and args.checkpointing_steps == "best":
                    if save_checkpoint:
                        accelerator.save_state(f"{args.output_dir}/best")
                    if (epoch + 1) % args.save_every == 0:
                        accelerator.save_state(f"{args.output_dir}/epoch_{epoch + 1}")

                if accelerator.is_main_process and args.checkpointing_steps == "epoch":
                    accelerator.save_state(f"{args.output_dir}/epoch_{epoch + 1}")

                model_saved = True
                logger.info("Model successfully saved.")

            except:
                logger.info("Save model failed. Retrying.")


if __name__ == "__main__":
    main()
