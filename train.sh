train_file="data/train_audiocaps.json"
validation_file="data/valid_audiocaps.json"
test_file="data/test_audiocaps_subset.json"
text_encoder_name="google/flan-t5-large"
scheduler_name="stabilityai/stable-diffusion-2-1"

# Original setting (from TANGO paper)
# unet_model_config="configs/tango_diffusion.json"
# tango_model="ckpt/declare-lab/tango-full-ft-audiocaps.pth"
# Lightweight setting (our lightweight TANGO)
unet_model_config="configs/tango_diffusion_light.json"
tango_model="saved/LightweightLDM/best/pytorch_model_2.bin"

stage1_model="saved/Stage1_variable_w/best/pytorch_model_2.bin"
stage2_model="saved/Stage2_variable_w/epoch_60/pytorch_model_2.bin"

# Train ConsistencyTTA with a flan-t5-large text encoder and mixed precision
# Stage 1 (Optional) -- Distill into a variable-guidance latent diffusion model
CUDA_VISIBLE_DEVICES="0, 1" accelerate launch train.py --stage 1 \
--train_file=$train_file --validation_file=$validation_file --test_file=$test_file \
--scheduler_name=$scheduler_name --text_encoder_name=$text_encoder_name --freeze_text_encoder \
--unet_model_config=$unet_model_config --tango_model=$tango_model \
--gradient_accumulation_steps=8 --per_device_train_batch_size=4 --per_device_eval_batch_size=6 \
--augment --num_train_epochs=50 --teacher_guidance_scale=-1 --text_column=captions --audio_column=location \
--target_ema_decay=.95 --ema_decay=.999 --learning_rate=1e-4 --adam_weight_decay=0 \
--checkpointing_steps=best --num_diffusion_steps=18 --num_warmup_steps=900 --use_bf16 --snr_gamma 5

# Stage 2 -- Distill into a CFG-Aware Latent-Space Consistency TTA Model
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py --stage 2 \
--train_file=$train_file --validation_file=$validation_file --test_file=$test_file \
--scheduler_name=$scheduler_name --text_encoder_name=$text_encoder_name --freeze_text_encoder \
--unet_model_config=$unet_model_config --tango_model=$tango_model --stage1_model=$stage1_model \
--gradient_accumulation_steps=5 --per_device_train_batch_size=6 --per_device_eval_batch_size=8 \
--augment --num_train_epochs=60 --teacher_guidance_scale=-1 --text_column=captions --audio_column=location \
--target_ema_decay=.95 --ema_decay=.999 --learning_rate=1e-5 --adam_weight_decay=1e-4 --use_edm --use_bf16 \
--checkpointing_steps=best --num_diffusion_steps=18 --num_warmup_steps=750 --snr_gamma 5 --loss_type "mse"

# Stage 3 (Optional) -- Finetune by maximizing CLAP score.
accelerate launch train.py --stage 2 \  # Stage 2 is not a typo.
--train_file=$train_file --validation_file=$validation_file --test_file=$test_file \
--scheduler_name=$scheduler_name --text_encoder_name=$text_encoder_name --freeze_text_encoder \
--unet_model_config=$unet_model_config --tango_model=$tango_model --stage1_model=$stage2_model \
--gradient_accumulation_steps=15 --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --seed=0 \
--augment --num_train_epochs=10 --teacher_guidance_scale=-1 --text_column=captions --audio_column=location \
--target_ema_decay=.95 --ema_decay=.999 --learning_rate=1e-6 --adam_weight_decay=1e-4 --use_edm --use_bf16 \
--checkpointing_steps=best --num_diffusion_steps=18 --num_warmup_steps=250 --snr_gamma 5 --loss_type "clap"

# To resume from a checkpoint, use:
# --resume_from_checkpoint="saved/(name)/(epoch)" --starting_epoch=0
