from pathlib import Path
from tqdm import tqdm
from typing import List, Union

from mamkit.data.processing import ProcessorComponent
from mm_argfallacy.utils.audio import load_audio_waveform


class AudioWaveformProcessor(ProcessorComponent):
    def __init__(
        self,
        target_sr: int,
    ):
        """
        Component that loads audio and context audio as waveform arrays.

        Args:
            target_sr: Target sampling rate to resample audio to.
        """
        self.target_sr = target_sr

    def __call__(
        self,
        inputs: List[Union[Path, List[Path]]],
        context: List[List[Path]] = None,
    ):
        """
        Loads and processes input and context audio files.
        """
        if context is None:
            context = [None] * len(inputs)

        input_features = []
        context_features = []

        for audio_input, context_input in tqdm(
            zip(inputs, context),
            desc="Loading Audio...",
            total=len(inputs),
        ):
            input_waveform = load_audio_waveform(audio_input, self.target_sr)
            context_waveform = load_audio_waveform(context_input, self.target_sr)

            input_features.append(input_waveform)

            if context_waveform is not None:
                context_features.append(context_waveform)

        return input_features, context_features or None
