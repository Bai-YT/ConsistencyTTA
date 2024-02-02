import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from laion_clap import CLAP_Module

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from audioldm_eval.datasets.load_mel import MelPairedDataset, WaveDataset
from audioldm_eval.metrics.fad import FrechetAudioDistance
from audioldm_eval import calculate_fid, calculate_isc, calculate_kid, calculate_kl
from audioldm_eval.feature_extractors.panns import Cnn14
from audioldm_eval.audio.tools import write_json
import audioldm_eval.audio as Audio

from ssr_eval.metrics import AudioMetrics
from tools.t2a_dataset import T2APairedDataset
from tools.torch_tools import seed_all


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def get_clap_features(pairedtextloader, clap_model):
    gt_features, gen_features, text_features, gen_mel_features = [], [], [], []

    for captions, gt_waves, gen_waves, gen_mels in tqdm(pairedtextloader):
        gt_waves = gt_waves.squeeze(1).float().to(device)
        gen_waves = gen_waves.squeeze(1).float().to(device)

        with torch.no_grad():
            seed_all(0)
            gt_features += [clap_model.get_audio_embedding_from_data(
                x=gt_waves, use_tensor=True
            )]
            seed_all(0)
            gen_features += [clap_model.get_audio_embedding_from_data(
                x=gen_waves, use_tensor=True
            )]
            seed_all(0)
            text_features += [clap_model.get_text_embedding(
                captions, use_tensor=True
            )]
            # TODO: get embedding from mel

    gt_features = torch.cat(gt_features, dim=0)
    gen_features = torch.cat(gen_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    return gt_features, gen_features, text_features


class EvaluationHelper:
    def __init__(self, sampling_rate, device, backbone="cnn14") -> None:

        self.device = device
        self.backbone = backbone
        self.sampling_rate = sampling_rate
        self.frechet = FrechetAudioDistance(use_pca=False, use_activation=False)
        self.lsd_metric = AudioMetrics(self.sampling_rate)
        self.frechet.model = self.frechet.model.to(self.device)

        features_list = ["2048", "logits"]
        if self.sampling_rate == 16000:
            self.mel_model = Cnn14(
                features_list=features_list,
                sample_rate=16000, window_size=512, hop_size=160,
                mel_bins=64, fmin=50, fmax=8000, classes_num=527,
            ).to(self.device)
        elif self.sampling_rate == 32000:
            self.mel_model = Cnn14(
                features_list=features_list,
                sample_rate=32000, window_size=1024, hop_size=320,
                mel_bins=64, fmin=50, fmax=14000, classes_num=527,
            ).to(self.device)
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rates."
            )
        self.mel_model.eval()
        self.fbin_mean, self.fbin_std = None, None

        if self.sampling_rate == 16000:
            self._stft = Audio.TacotronSTFT(
                filter_length=512, hop_length=160, win_length=512,
                n_mel_channels=64, sampling_rate=16000, mel_fmin=50, mel_fmax=8000
            )
        elif self.sampling_rate == 32000:
            self._stft = Audio.TacotronSTFT(
                filter_length=1024, hop_length=320, win_length=1024,
                n_mel_channels=64, sampling_rate=32000, mel_fmin=50, mel_fmax=14000
            )
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rates."
            )

        self.clap_model = CLAP_Module(enable_fusion=False, amodel='HTSAT-base').to(device)
        self.clap_model.load_ckpt(
            'ckpt/music_audioset_epoch_15_esc_90.14.pt', verbose=False
        )

    def file_init_check(self, dir):
        assert os.path.exists(dir), "The path does not exist %s" % dir
        assert len(os.listdir(dir)) > 1, "There is no files in %s" % dir

    def get_filename_intersection_ratio(
        self, dir1, dir2, threshold=0.99, limit_num=None
    ):
        self.datalist1 = [os.path.join(dir1, x) for x in os.listdir(dir1)]
        self.datalist1 = sorted(self.datalist1)
        self.datalist1 = [item for item in self.datalist1 if item.endswith(".wav")]

        self.datalist2 = [os.path.join(dir2, x) for x in os.listdir(dir2)]
        self.datalist2 = sorted(self.datalist2)
        self.datalist2 = [item for item in self.datalist2 if item.endswith(".wav")]

        data_dict1 = {os.path.basename(x): x for x in self.datalist1}
        data_dict2 = {os.path.basename(x): x for x in self.datalist2}

        keyset1 = set(data_dict1.keys())
        keyset2 = set(data_dict2.keys())

        intersect_keys = keyset1.intersection(keyset2)
        if len(intersect_keys) / len(keyset1) > threshold \
            and len(intersect_keys) / len(keyset2) > threshold:
            return True
        else:
            return False

    def calculate_lsd(self, pairedloader, same_name=True, time_offset=160 * 7):
        if same_name == False:
            return {"lsd": -1, "ssim_stft": -1}

        lsd_avg = []
        ssim_stft_avg = []
        for _, _, _, (audio1, audio2) in tqdm(pairedloader):
            audio1, audio2 = audio1.numpy()[0, :], audio2.numpy()[0, :]

            # HIFIGAN (verified on 2023-01-12) requires seven frames' offset
            audio1 = audio1[time_offset:]
            audio1 = (audio1 - audio1.mean()) / np.abs(audio1).max()
            audio2 = (audio2 - audio2.mean()) / np.abs(audio2).max()

            min_len = min(audio1.shape[0], audio2.shape[0])
            audio1, audio2 = audio1[:min_len], audio2[:min_len]

            result = self.lsd(audio1, audio2)
            lsd_avg.append(result["lsd"])
            ssim_stft_avg.append(result["ssim"])

        return {"lsd": np.mean(lsd_avg), "ssim_stft": np.mean(ssim_stft_avg)}

    def lsd(self, audio1, audio2):
        result = self.lsd_metric.evaluation(audio1, audio2, None)
        return result

    def calculate_psnr_ssim(self, pairedloader, same_name=True):
        if same_name == False:
            return {"psnr": -1, "ssim": -1}
        psnr_avg, ssim_avg = [], []

        for mel_gen, mel_target, filename, _ in tqdm(pairedloader):
            mel_gen = mel_gen.cpu().numpy()[0]
            mel_target = mel_target.cpu().numpy()[0]
            psnrval = psnr(mel_gen, mel_target)
            if np.isinf(psnrval):
                print("Infinite value encountered in psnr %s " % filename)
                continue
            psnr_avg.append(psnrval)
            ssim_avg.append(ssim(mel_gen, mel_target, data_range=1.))

        return {"psnr": np.mean(psnr_avg), "ssim": np.mean(ssim_avg)}

    def calculate_metrics(
        self, dataset_json_path, generate_files_path, groundtruth_path,
        mel_path, same_name, target_length=1000, limit_num=None
    ):
        # Generation, target
        seed_all(0)
        print(f"generate_files_path: {generate_files_path}")
        print(f"groundtruth_path: {groundtruth_path}")

        # Check if the generated directory and the groundtruth directory have same files
        print("Checking file integrity...")
        generated_files = [f for f in os.listdir(generate_files_path) if f.endswith('.wav')]
        groundtruth_files = [f for f in os.listdir(groundtruth_path) if f.endswith('.wav')]
        if generated_files != groundtruth_files:
            print("In generated but not in groundtruth: "
                  f"{set(generated_files) - set(groundtruth_files)}")
            print("In groundtruth but not in generated: "
                  f"{set(groundtruth_files) - set(generated_files)}")
            raise ValueError(
                "Generated and groundtruth diretories have different files.\n"
                f"Generated: {generated_files};\nGround truth: {groundtruth_files}."
            )
        print("Done checking files.")

        outputloader, resultloader = tuple([
            DataLoader(
                WaveDataset(
                    path, self.sampling_rate, limit_num=limit_num, target_length=tl
                ), batch_size=1, sampler=None, num_workers=4
            ) for path, tl in zip([generate_files_path, groundtruth_path], [target_length, 1000])
        ])

        pairedtextdataset = T2APairedDataset(
            dataset_json_path=dataset_json_path, generated_path=generate_files_path,
            target_length=target_length, mel_path=mel_path, sample_rate=[16000, 48000]
        )
        pairedtextloader = DataLoader(
            pairedtextdataset, batch_size=16, num_workers=8, shuffle=False,
            collate_fn=pairedtextdataset.collate_fn
        )

        melpaireddataset = MelPairedDataset(
            generate_files_path, groundtruth_path, self._stft, self.sampling_rate,
            self.fbin_mean, self.fbin_std, limit_num=limit_num,
        )
        melpairedloader = DataLoader(
            melpaireddataset, batch_size=1, sampler=None, num_workers=8, shuffle=False
        )

        out = {}

        print("Calculating Frechet Audio Distance...")  # Frechet audio distance
        fad_score = self.frechet.score(
            generate_files_path, groundtruth_path, target_length=target_length
        )
        out.update(fad_score)

        print("Calculating LSD...")  # LSD
        metric_lsd = self.calculate_lsd(melpairedloader, same_name=same_name)
        out.update(metric_lsd)

        # Get CLAP features
        print("Calculating CLAP score...")  # CLAP Score
        gt_feat, gen_feat, text_feat = get_clap_features(pairedtextloader, self.clap_model)
        # CLAP similarity calculation
        gt_text_similarity = cosine_similarity(gt_feat, text_feat, dim=1)
        gen_text_similarity = cosine_similarity(gen_feat, text_feat, dim=1)
        gen_gt_similarity = cosine_similarity(gen_feat, gt_feat, dim=1)
        gt_text_similarity = torch.clamp(gt_text_similarity, min=0)
        gen_text_similarity = torch.clamp(gen_text_similarity, min=0)
        gen_gt_similarity = torch.clamp(gen_gt_similarity, min=0)
        # Update output dict
        out.update({
            'gt_text_clap_score': gt_text_similarity.mean().item() * 100.,
            'gen_text_clap_score': gen_text_similarity.mean().item() * 100.,
            'gen_gt_clap_score': gen_gt_similarity.mean().item() * 100.
        })

        print("Calculating PSNR...")  # PSNR and SSIM
        metric_psnr_ssim = self.calculate_psnr_ssim(
            melpairedloader, same_name=same_name
        )
        out.update(metric_psnr_ssim)

        print("Getting Mel features...")  # Get Mel features
        featuresdict_2 = self.get_featuresdict(resultloader)
        featuresdict_1 = self.get_featuresdict(outputloader)

        print("Calculating KL divergence...")  # KL divergence
        metric_kl, kl_ref, paths_1 = calculate_kl(
            featuresdict_1, featuresdict_2, "logits", same_name
        )
        out.update(metric_kl)

        print("Calculating inception score...")  # Inception Score
        metric_isc = calculate_isc(
            featuresdict_1, feat_layer_name="logits", splits=10,
            samples_shuffle=True, rng_seed=2020,
        )
        out.update(metric_isc)

        print("Calculating kernel inception distance...")  # Kernel Inception Distance
        metric_kid = calculate_kid(
            featuresdict_1, featuresdict_2, feat_layer_name="2048", degree=3, gamma=None,
            subsets=100, subset_size=len(pairedtextdataset), coef0=1, rng_seed=2020,
        )
        out.update(metric_kid)

        print("Calculating Frechet distance...")  # Frechet Distance
        metric_fid = calculate_fid(
            featuresdict_1, featuresdict_2, feat_layer_name="2048"
        )
        out.update(metric_fid)

        keys_list = [
            "frechet_distance", "frechet_audio_distance", "lsd", "psnr",
            "kullback_leibler_divergence_sigmoid", "kullback_leibler_divergence_softmax",
            "ssim", "ssim_stft", "inception_score_mean", "inception_score_std",
            "kernel_inception_distance_mean", "kernel_inception_distance_std",
            "gt_text_clap_score", "gen_text_clap_score", "gen_gt_clap_score"
        ]
        result = {}
        for key in keys_list:
            result[key] = round(out.get(key, float("nan")), 4)

        json_path = generate_files_path + "_evaluation_results.json"
        write_json(result, json_path)
        return result

    def get_featuresdict(self, dataloader):
        out, out_meta = None, None

        for waveform, filename in tqdm(dataloader):
            metadict = {"file_path_": filename}
            waveform = waveform.squeeze(1).float().to(self.device)

            with torch.no_grad():
                featuresdict = self.mel_model(waveform)
                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            out = featuresdict if out is None else {
                k: out[k] + featuresdict[k] for k in out.keys()
            }
            out_meta = metadict if out_meta is None else {
                k: out_meta[k] + metadict[k] for k in out_meta.keys()
            }

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}

    def sample_from(self, samples, number_to_use):
        assert samples.shape[0] >= number_to_use
        rand_order = np.random.permutation(samples.shape[0])
        return samples[rand_order[: samples.shape[0]], :]

    def main(
        self, dataset_json_path, generated_files_path, groundtruth_path,
        mel_path=None, target_length=1000, limit_num=None,
    ):
        self.file_init_check(generated_files_path)
        self.file_init_check(groundtruth_path)

        same_name = self.get_filename_intersection_ratio(
            generated_files_path, groundtruth_path, limit_num=limit_num
        )
        return self.calculate_metrics(
            dataset_json_path, generated_files_path, groundtruth_path,
            mel_path, same_name, target_length, limit_num
        )
