"""
Calculate Frechet Audio Distance betweeen two audio directories.
Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid
VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import resampy


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_audio_task(fname, target_sr=16000, target_length=1000):
    wav_data, orig_sr = sf.read(fname, dtype="int16")
    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    wav_data = wav_data / 32768.  # Convert to [-1.0, +1.0]

    # Convert to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    if orig_sr % target_sr == 0:
        wav_data = wav_data[::(orig_sr // target_sr)]
    elif orig_sr != target_sr:
        wav_data = resampy.resample(wav_data, orig_sr, target_sr, filter="kaiser_best")

    return wav_data[:int(target_length * target_sr / 100)]


class FrechetAudioDistance:
    def __init__(
        self, use_pca=False, use_activation=False, audio_load_worker=8
    ):
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.audio_load_worker = audio_load_worker

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either
            (i)  a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()

    def get_embeddings(self, audio_paths, sr=16000, target_length=1000):
        """
        Get embeddings using VGGish model.
        Params:
        -- audio_paths  :   A list of np.ndarray audio samples
        -- sr           :   Sampling rate. Default value is 16000.
        -- target_length:   Target audio length in centiseconds.
        """
        embd_lst = []
        for _, fname in enumerate(tqdm(os.listdir(audio_paths))):
            if fname.endswith(".wav"):
                audio = load_audio_task(
                    os.path.join(audio_paths, fname), target_sr=sr, target_length=target_length
                )
                embd = self.model.forward(audio, sr).cpu().detach().numpy()
                embd_lst.append(embd)

        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: 
        https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def score(self, generated_dir, groundtruth_dir, target_length=1000, store_embds=False):

        generated_embds = self.get_embeddings(generated_dir, target_length=target_length)
        groundtruth_embds = self.get_embeddings(groundtruth_dir, target_length=1000)

        if store_embds:
            np.save("generated_embds.npy", generated_embds)
            np.save("groundtruth_embds.npy", groundtruth_embds)

        if len(generated_embds) == 0:
            print("[Frechet Audio Distance] generated dir is empty, exitting...")
            return -1
        if len(groundtruth_embds) == 0:
            print("[Frechet Audio Distance] ground truth dir is empty, exitting...")
            return -1

        groundtruth_mu, groundtruth_sigma = self.calculate_embd_statistics(groundtruth_embds)
        generated_mu, generated_sigma = self.calculate_embd_statistics(generated_embds)

        fad_score = self.calculate_frechet_distance(
            generated_mu, generated_sigma, groundtruth_mu, groundtruth_sigma
        )
        return {"frechet_audio_distance": fad_score}
