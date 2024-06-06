import torch
import soundfile as sf
import numpy as np
import random, os

from consistencytta import ConsistencyTTA


def seed_all(seed):
    """ Seed all random number generators. """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate(prompt: str, seed: int = None, cfg_weight: float = 4.):
    """ Generate audio from a given prompt.
    Args:
        prompt (str): Text prompt to generate audio from.
        seed (int, optional): Random seed. Defaults to None, which means no seed.
    """
    if seed is not None:
        seed_all(seed)

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            wav = consistencytta(
                [prompt], num_steps=1, cfg_scale_input=cfg_weight, cfg_scale_post=1., sr=sr
            )
        sf.write("output.wav", wav.T, samplerate=sr, subtype='PCM_16')

    return "output.wav"


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)
sr = 16000

# Build ConsistencyTTA model
consistencytta = ConsistencyTTA().to(device)
consistencytta.eval()
consistencytta.requires_grad_(False)

# Generate audio (feel free to change the seed and prompt)
print("Generating test audio for prompt 'A dog barks as a train passes by.'...")
generate("A dog barks as a train passes by.", seed=1)

while True:
    prompt = input("Enter a prompt: ")
    generate(prompt)
    print(f"Audio generated successfully for prompt {prompt}!")
