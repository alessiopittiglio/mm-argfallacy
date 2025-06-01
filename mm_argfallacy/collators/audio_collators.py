from typing import Dict, Optional
from transformers import AutoProcessor, AutoFeatureExtractor


class AudioCollator:
    """
    Collator for processing waveform audio inputs with a transformer processor.
    """

    def __init__(
        self,
        model_card: str,
        target_sr: int,
        max_length_seconds: Optional[float] = None,
        processor_args: Optional[Dict] = None,
    ):

        self.target_sr = target_sr
        self.processor_args = processor_args or {}

        if "hubert" in model_card:
            self.processor = AutoFeatureExtractor.from_pretrained(model_card)
        else:
            self.processor = AutoProcessor.from_pretrained(model_card)

        if max_length_seconds and max_length_seconds > 0:
            self.processor_args.update(
                {
                    "max_length": int(target_sr * max_length_seconds),
                    "truncation": True,
                }
            )

    def _process_waveforms(self, waveforms):
        return self.processor(
            waveforms,
            sampling_rate=self.target_sr,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            **self.processor_args
        )

    def __call__(self, inputs, context=None) -> Dict:
        processed_inputs = self._process_waveforms(inputs)

        if context is not None:
            processed_context = self._process_waveforms(context)

        return {
            "inputs": processed_inputs.input_values,
            "input_mask": processed_inputs.attention_mask,
            "context": processed_context.input_values,
            "context_mask": processed_context.attention_mask,
        }
