import pickle
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture


class LFCCGMMBaseline:
    """
    Baseline B0: LFCC-GMM for Voice Anti-Spoofing.
    Two GMMs trained independently:
    - M_bonafide on bona fide speech LFCC features
    - M_spoof on spoofed speech LFCC features

    Score for utterance with frame features X:
    score(X) = (1/T) * [ log P(X | M_bonafide) - log P(X | M_spoof) ]
    """

    def __init__(self, n_components: int = 512, max_iter: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

        self.gmm_bonafide: GaussianMixture | None = None
        self.gmm_spoof: GaussianMixture | None = None

    def fit_bonafide(self, features: np.ndarray):
        """Fit GMM on all concatenated frames from bonafide utterances. shape: (N_frames, 60)"""
        print(f"Fitting GMM Bonafide ({self.n_components} components on {features.shape[0]} frames)...")
        self.gmm_bonafide = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm_bonafide.fit(features)
        print("GMM Bonafide fitted successfully.")

    def fit_spoof(self, features: np.ndarray):
        """Fit GMM on all concatenated frames from spoof utterances. shape: (N_frames, 60)"""
        print(f"Fitting GMM Spoof ({self.n_components} components on {features.shape[0]} frames)...")
        self.gmm_spoof = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm_spoof.fit(features)
        print("GMM Spoof fitted successfully.")

    def score_utterance(self, x: np.ndarray) -> float:
        """
        Compute Log-Likelihood Ratio (LLR) score for single utterance.
        x: (T, 60)
        """
        if self.gmm_bonafide is None or self.gmm_spoof is None:
            raise RuntimeError("Both GMM models must be fitted before scoring.")

        # per-frame log probabilities
        llk_bonafide = self.gmm_bonafide.score_samples(x)
        llk_spoof = self.gmm_spoof.score_samples(x)

        # Average log-likelihood ratio across all frames in utterance
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
        print(f"Baseline model loaded from {filepath}")
