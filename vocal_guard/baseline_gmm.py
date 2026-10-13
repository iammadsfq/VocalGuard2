import math
import pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.mixture import GaussianMixture


class TorchGMM(torch.nn.Module):
    """
    GPU-Accelerated Gaussian Mixture Model Evaluator (PyTorch CUDA).
    Evaluates log P(X | M) in parallel on GPU.
    """

    def __init__(self, sklearn_gmm: GaussianMixture, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights = torch.tensor(sklearn_gmm.weights_, dtype=torch.float32, device=self.device)
        self.means = torch.tensor(sklearn_gmm.means_, dtype=torch.float32, device=self.device)  # (K, D)
        self.covars = torch.tensor(sklearn_gmm.covariances_, dtype=torch.float32, device=self.device)  # (K, D)
        self.log_weights = torch.log(self.weights)
        self.log_det = torch.log(self.covars).sum(dim=-1)
        self.D = self.means.shape[-1]
        self.const = self.D * math.log(2.0 * math.pi)

    def score_samples(self, X_t: torch.Tensor) -> torch.Tensor:
        """
        X_t: (N, D) on self.device
        Returns: (N,) log probability per frame
        """
        diff = X_t.unsqueeze(1) - self.means.unsqueeze(0)  # (N, K, D)
        quad = torch.sum((diff**2) / self.covars.unsqueeze(0), dim=-1)  # (N, K)
        log_p_k = -0.5 * (self.const + self.log_det.unsqueeze(0) + quad) + self.log_weights.unsqueeze(0)
        return torch.logsumexp(log_p_k, dim=-1)

    def score_utterance(self, X_t: torch.Tensor) -> float:
        """Mean log-likelihood across all frames in an utterance."""
        return float(self.score_samples(X_t).mean().item())


class LFCCGMMBaseline:
    """
    Baseline B0: LFCC-GMM for Voice Anti-Spoofing.
    Two GMMs trained independently:
    - M_bonafide on bona fide speech LFCC features
    - M_spoof on spoofed speech LFCC features

    Score for utterance with frame features X:
    score(X) = (1/T) * [ log P(X | M_bonafide) - log P(X | M_spoof) ]
    """

    def __init__(self, n_components: int = 256, max_iter: int = 100, random_state: int = 42, device: str = "cuda"):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device

        self.gmm_bonafide: GaussianMixture | None = None
        self.gmm_spoof: GaussianMixture | None = None
        self.torch_gmm_bonafide: TorchGMM | None = None
        self.torch_gmm_spoof: TorchGMM | None = None

    def fit_bonafide(self, features: np.ndarray):
        """Fit GMM on bonafide frames."""
        print(f"Fitting GMM Bonafide ({self.n_components} components on {features.shape[0]} frames)...")
        self.gmm_bonafide = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm_bonafide.fit(features)
        self.torch_gmm_bonafide = TorchGMM(self.gmm_bonafide, device=self.device)
        print("GMM Bonafide fitted and transferred to GPU.")

    def fit_spoof(self, features: np.ndarray):
        """Fit GMM on spoof frames."""
        print(f"Fitting GMM Spoof ({self.n_components} components on {features.shape[0]} frames)...")
        self.gmm_spoof = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm_spoof.fit(features)
        self.torch_gmm_spoof = TorchGMM(self.gmm_spoof, device=self.device)
        print("GMM Spoof fitted and transferred to GPU.")

    def score_utterance(self, x: np.ndarray) -> float:
        """Compute LLR score for an utterance (accelerated on GPU if available)."""
        if self.torch_gmm_bonafide is not None and self.torch_gmm_spoof is not None:
            dev = self.torch_gmm_bonafide.device
            x_t = torch.tensor(x, dtype=torch.float32, device=dev)
            llk_bona = self.torch_gmm_bonafide.score_utterance(x_t)
            llk_spoof = self.torch_gmm_spoof.score_utterance(x_t)
            return llk_bona - llk_spoof

        if self.gmm_bonafide is None or self.gmm_spoof is None:
            raise RuntimeError("Both GMM models must be fitted before scoring.")

        llk_bonafide = self.gmm_bonafide.score_samples(x)
        llk_spoof = self.gmm_spoof.score_samples(x)
        return float(np.mean(llk_bonafide - llk_spoof))

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"bonafide": self.gmm_bonafide, "spoof": self.gmm_spoof}, f)
        print(f"Baseline model saved to {filepath}")

    def load(self, filepath: str | Path):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.gmm_bonafide = data["bonafide"]
            self.gmm_spoof = data["spoof"]
            self.torch_gmm_bonafide = TorchGMM(self.gmm_bonafide, device=self.device)
            self.torch_gmm_spoof = TorchGMM(self.gmm_spoof, device=self.device)
        print(f"Baseline model loaded from {filepath}")
