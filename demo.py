import os
import json
from time import time
import argparse
import soundfile as sf
import numpy as np
import torch

from diffusers import DDIMScheduler, HeunDiscreteScheduler
from models import AudioLCM, AudioLCM_FTVAE
from tools.build_pretrained import build_pretrained_models
from tools.torch_tools import seed_all
from inference import dotdict


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for text to audio generation task."
    )
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path for saved model bin file."
    )
    parser.add_argument(
        "--num_teacher_steps", type=int, default=80,
        help="How many teacher denoising steps for generation."
    )
    parser.add_argument(
        "--cfg_weight", type=int, default=4.0,
        help="Classifier-free guidance weight for the student model."
    )
    parser.add_argument(
        "--use_ema", action="store_true", default=False,
        help="Use the EMA model for inference."
    )
    parser.add_argument(
        "--use_edm", action="store_true", default=False,
        help="Use EDM's solver and scheduler."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    sr = 16000

    name = "audioldm-s-full"
    vae, stft = build_pretrained_models(name)
    vae, stft = vae.to(device), stft.to(device)

    if train_args.finetune_vae:
        model_class = AudioLCM_FTVAE
    else:
        model_class = AudioLCM

    model = model_class(
        text_encoder_name=train_args.text_encoder_name,
        scheduler_name=train_args.scheduler_name,
        unet_model_name=train_args.unet_model_name,
        unet_model_config_path=train_args.unet_model_config,
        snr_gamma=train_args.snr_gamma,
        freeze_text_encoder=train_args.freeze_text_encoder,
        uncondition=train_args.uncondition,  # only effective for stage-2
        use_edm=train_args.use_edm,  # only effective for stage-2
        use_karras=train_args.use_karras,  # only effective for stage-2
        use_lora=train_args.use_lora,
        target_ema_decay=train_args.target_ema_decay,
        ema_decay=train_args.ema_decay,
        num_diffusion_steps=train_args.num_diffusion_steps,
        teacher_guidance_scale=train_args.teacher_guidance_scale,
        loss_type=train_args.loss_type,
        vae=vae
    ).to(device)

    # Load Trained Weight
    model.load_pretrained(torch.load(args.model, map_location='cpu'))
    model.eval()

    sched_class = HeunDiscreteScheduler if args.use_edm else DDIMScheduler
    scheduler = sched_class.from_pretrained(
        train_args.scheduler_name, subfolder="scheduler"
    )

    # Prompt the user to give the textual prompt
    cntr = 0
    while True:
        if args.seed is not None:
            seed_all(args.seed)

        texts = []
        while True:
            text = input("Please enter textual prompt: ")
            if text == "" and len(texts) > 0:
                break
            elif text != "":
                texts += [text]

        with torch.no_grad():
            zhat_0_stu, zhat_0_tea, time_stu, time_tea = model.inference(
                texts, scheduler, query_teacher=True, return_all=True,
                num_steps=1, num_teacher_steps=args.num_teacher_steps,
                guidance_scale_input=args.cfg_weight, guidance_scale_post=1.,
                use_edm=args.use_edm, use_ema=args.use_ema
            )

            t_start_post_process = time()
            zhat_0 = torch.cat([zhat_0_stu, zhat_0_tea], dim=0)
            mel_all = vae.decode_first_stage(zhat_0.float())
            wav_all = vae.decode_to_waveform(mel_all)[:, :int(sr * 10)]  # Truncate to 10 seconds
            wav_stu, wav_tea = np.split(wav_all, 2)
            time_post_process = time() - t_start_post_process

        # Save
        for w_stu, w_tea in zip(wav_stu, wav_tea):
            os.makedirs("demo_outputs", exist_ok=True)
            sf.write(f"demo_outputs/distilled_{cntr}.wav", w_stu, samplerate=sr)
            sf.write(f"demo_outputs/diffusion_{cntr}.wav", w_tea, samplerate=sr)
            cntr += 1

        time_stu += time_post_process / 2
        time_tea += time_post_process / 2
        print(f"Generation time for the distilled model: {time_stu} seconds.")
        print(f"Generation time for the diffusion baseline: {time_tea} seconds.")


if __name__ == "__main__":
    main()
