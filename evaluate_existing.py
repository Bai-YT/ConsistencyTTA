import os
import argparse
import torch
import soundfile as sf

from tools.torch_tools import seed_all
from audioldm_eval import EvaluationHelper


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for text to audio generation task."
    )
    parser.add_argument(
        "--dataset_json_path", type=str, help="Path to the test dataset json file.",
        default="data/test_audiocaps_subset.json"
    )
    parser.add_argument(
        "--gen_files_path", required=True, type=str,
        help="Path to the folder that contains the generated files."
    )
    parser.add_argument(
        '--test_references', type=str, help="Path to the test dataset json file.",
        default="dataset/audiocaps_test_references/subset"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Random seed. Default to 0."
    )
    parser.add_argument(
        '--target_length', default=970, type=int,
        help="Audio truncation length (in centiseconds)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    all_outputs = []
    for filename in os.listdir(args.gen_files_path):
        if filename.endswith(".wav"):
            wav_tuple = sf.read(os.path.join(args.gen_files_path, filename))
            all_outputs += [wav_tuple[0]]

    evaluator = EvaluationHelper(sampling_rate=16000, device=device)

    result = evaluator.main(
        dataset_json_path=args.dataset_json_path, generated_files_path=args.gen_files_path,
        groundtruth_path=args.test_references, target_length=args.target_length
    )
    result["Test Instances"] = len(all_outputs)
    print(result)


if __name__ == "__main__":
    main()
