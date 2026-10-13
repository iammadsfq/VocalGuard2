from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class ProtocolEntry:
    speaker_id: str
    audio_id: str
    system_id: str
    key_id: str
    label: str  # 'bonafide' or 'spoof'

    @property
    def is_bonafide(self) -> bool:
        return self.label.lower() == "bonafide"


def read_protocol_file(file_path: str | Path) -> pd.DataFrame:
    """
    Parse official ASVspoof 2019 LA protocol file.
    Columns: speaker_id, audio_id, system_id, attack_id, label
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {file_path}")

    # Read space-delimited protocol file
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "audio_id", "system_id", "attack_id", "label"],
        dtype=str,
    )
    df["label"] = df["label"].str.lower()
    return df
