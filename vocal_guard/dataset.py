import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from vocal_guard.features import extract_lfcc


def pad_or_truncate(audio: np.ndarray, target_length: int = 64000) -> np.ndarray:
    """
    Standard ASVspoof audio pre-processing:
    - If shorter than target_length (e.g. 4 seconds @ 16kHz = 64,000 samples), repeat (wrap) audio.
    - If longer, truncate to target_length.
    """
    length = len(audio)
    if length == target_length:
        return audio
    if length > target_length:
        return audio[:target_length]
    # Wrap repeat
    repeat_factor = int(np.ceil(target_length / length))
    repeated = np.tile(audio, repeat_factor)
    return repeated[:target_length]


class ASVspoof2019Dataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 LA partition.

    Extracts 60-dimensional LFCC (static + delta + delta-delta).
    Returns:
        feature: torch.FloatTensor of shape (1, 60, T)
        label: torch.LongTensor scalar (0 for bonafide, 1 for spoof)
        utt_id: str
    """

    def __init__(
        self,
        protocol_df: pd.DataFrame,
        audio_dir: str,
        target_length: int = 64000,
        sr: int = 16000,
    ):
        self.df = protocol_df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.sr = sr

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        utt_id = row["audio_id"]
        label_str = row["label"]
        file_path = f"{self.audio_dir}/{utt_id}.flac"

        try:
            audio, sr = sf.read(file_path)
            if sr != self.sr:
                raise ValueError(f"Sample rate mismatch: expected {self.sr}, got {sr}")
        except Exception:
            # Fallback if file read fails (e.g. corrupted audio)
            audio = np.zeros(self.target_length, dtype=np.float32)

        # Pad or truncate to fixed 4 seconds
        audio = pad_or_truncate(audio, target_length=self.target_length)

        # Extract LFCC (T, 60)
        lfcc = extract_lfcc(audio, sr=self.sr, with_deltas=True)

        # Transpose to (60, T) then add channel dim -> (1, 60, T)
        feat = torch.from_numpy(lfcc.T).float().unsqueeze(0)

        # Label: bonafide -> 0, spoof -> 1
        label = 0 if label_str == "bonafide" else 1

        return feat, torch.tensor(label, dtype=torch.long), utt_id
