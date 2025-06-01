from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from mamkit.data.collators import UnimodalCollator
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from mamkit.data.processing import UnimodalProcessor

from mm_argfallacy.collators.text_collators import TextCollator, ConcatTextCollator
from mm_argfallacy.processing import AudioWaveformProcessor
from mm_argfallacy.collators.audio_collators import AudioCollator


class MMUSEDFallacyDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: dict,
        collator_cfg: dict,
        data_path: str = "data_old/",
        batch_size: int = 8,
        num_workers: int = 4,
        validation_split: float = 0.2,
        split_key: str = "default",
        seed: int = 20,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.collator_cfg = collator_cfg
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.split_key = split_key
        self.seed = seed

        self.input_mode = InputMode[self.dataset_cfg["input_mode"]]
        self.collator = self._build_collator()

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        cfg = self.dataset_cfg.copy()
        cfg["input_mode"] = InputMode[cfg["input_mode"]]

        dataset = MMUSEDFallacy(base_data_path=self.data_path, **cfg)
        split = dataset.get_splits(key=self.split_key)[0]

        split.train = self._remove_duplicates(split.train)

        if self.input_mode == InputMode.AUDIO_ONLY:
            if stage in ("fit", None):
                split.train = self._preprocess_audio(split.train)
            if stage in ("test", None):
                split.test = self._preprocess_audio(split.test)

        # Train/val split
        labels = split.train.labels
        indices = np.arange(len(split.train))

        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.validation_split,
            stratify=labels,
            random_state=self.seed,
        )

        if stage in ("fit", None):
            self.train_dataset = Subset(split.train, train_idx)
            self.val_dataset = Subset(split.train, val_idx)
            self.collator = self._build_collator()

        if stage in ("test", None):
            self.test_dataset = split.test
            self.test_collator = self._build_collator(label_collator=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        test_collator = self._build_collator(label_collator=None)

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=test_collator,
        )

    def _build_collator(
        self,
        label_collator=lambda labels: torch.tensor(labels),
    ) -> UnimodalCollator:
        name = self.collator_cfg["name"]

        if self.input_mode == InputMode.TEXT_ONLY:
            tokenizer_cfg = self.collator_cfg["tokenizer"]
            if name == "concat":
                features_collator = ConcatTextCollator(
                    model_card=tokenizer_cfg["model_card"],
                    tokenizer_args=tokenizer_cfg["params"],
                    context_first=False,
                )
            else:
                features_collator = TextCollator(
                    model_card=tokenizer_cfg["model_card"],
                    tokenizer_args=tokenizer_cfg["params"],
                )
        elif self.input_mode == InputMode.AUDIO_ONLY:
            processor_cfg = self.collator_cfg["processor"]
            params = processor_cfg["params"]
            features_collator = AudioCollator(
                model_card=processor_cfg["model_card"],
                target_sr=params["sampling_rate"],
                max_length_seconds=params["max_length_seconds"],
            )
        else:
            raise ValueError(f"Unsupported input mode: {self.input_mode}")

        return UnimodalCollator(
            features_collator=features_collator,
            label_collator=label_collator,
        )

    def _preprocess_audio(self, dataset):
        sampling_rate = self.collator_cfg["processor"]["params"]["sampling_rate"]
        features_processor = AudioWaveformProcessor(target_sr=sampling_rate)
        processor = UnimodalProcessor(features_processor=features_processor)
        return processor(dataset)

    def _remove_duplicates(self, dataset):
        # TODO: implement deduplication logic
        return dataset
