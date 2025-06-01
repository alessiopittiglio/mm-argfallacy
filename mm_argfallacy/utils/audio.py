from pathlib import Path
from typing import Union, List

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


def load_audio_waveform(
    audio_input: Union[None, Path, List[Path]],
    target_sr: int,
    min_duration_sec: float = 0.1,
) -> np.ndarray:
    """
    Load and preprocess audio waveform, resampling to target sample rate if needed.

    Args:
        audio_input: Path or list of Paths to audio files, or None.
        target_sr: Target sample rate.
        min_duration_sec: Minimum duration in seconds to return silence if input is
        invalid.

    Returns:
        Numpy array with audio waveform.
    """

    def empty_waveform() -> np.ndarray:
        return np.zeros(int(target_sr * min_duration_sec), dtype=np.float32)

    if audio_input is None or (isinstance(audio_input, list) and not audio_input):
        return empty_waveform()

    if isinstance(audio_input, Path):
        if not audio_input.exists():
            return empty_waveform()
        waveform, source_sr = torchaudio.load(audio_input)
    elif isinstance(audio_input, list):
        _, source_sr = torchaudio.load(audio_input[0])
        waveform = torch.cat([torchaudio.load(path)[0] for path in audio_input], dim=-1)
    else:
        raise ValueError("audio_input must be a Path or a list of Paths.")

    if source_sr != target_sr:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            waveform = F.resample(waveform, orig_freq=source_sr, new_freq=target_sr)
            waveform = waveform.squeeze(0)
        else:
            waveform = F.resample(waveform, orig_freq=source_sr, new_freq=target_sr)

    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    elif waveform.dim() > 1 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)

    return waveform.numpy()
