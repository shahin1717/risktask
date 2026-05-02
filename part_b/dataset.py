import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List


# ── Dataset ─────────────────────────────────────────────────────────────────

class AzerbaijaniASRDataset(Dataset):
    """
    PyTorch Dataset for Azerbaijani ASR fine-tuning with Whisper.
    Loads audio, extracts log-mel features and tokenizes reference text.
    """

    def __init__(self, hf_dataset, processor: WhisperProcessor, sample_rate: int = 16000):
        """
        Args:
            hf_dataset  : HuggingFace dataset split (train / validation / test)
            processor   : WhisperProcessor (feature extractor + tokenizer)
            sample_rate : Target sampling rate (Whisper expects 16kHz)
        """
        self.dataset     = hf_dataset
        self.processor   = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample     = self.dataset[idx]
        audio_path = sample["path"]
        reference  = sample["sentence"]

        # Load and resample audio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Extract log-mel features
        input_features = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features[0]   # shape: (80, 3000)

        # Tokenize reference text into label IDs
        labels = self.processor.tokenizer(reference).input_ids

        return {
            "input_features": input_features,
            "labels":         torch.tensor(labels, dtype=torch.long)
        }


# ── Data Collator ────────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    """
    Pads input_features and labels to same length within a batch.
    Replaces padding token IDs in labels with -100 so they are
    ignored by the loss function.
    """
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        # ── Pad input features ───────────────────────────────────────────────
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # ── Pad labels ───────────────────────────────────────────────────────
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding token ID with -100 (ignored in loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


# ── Factory functions ────────────────────────────────────────────────────────

def build_dataset(hf_dataset, processor: WhisperProcessor,
                  num_samples: int = None) -> AzerbaijaniASRDataset:
    """
    Build AzerbaijaniASRDataset, optionally limiting to num_samples.

    Args:
        hf_dataset  : Raw HuggingFace dataset split
        processor   : WhisperProcessor
        num_samples : If set, take only first N samples

    Returns:
        AzerbaijaniASRDataset
    """
    if num_samples is not None:
        hf_dataset = hf_dataset.select(range(min(num_samples, len(hf_dataset))))

    return AzerbaijaniASRDataset(hf_dataset, processor)


def build_collator(processor: WhisperProcessor) -> WhisperDataCollator:
    """
    Build WhisperDataCollator.

    Args:
        processor: WhisperProcessor

    Returns:
        WhisperDataCollator
    """
    return WhisperDataCollator(processor=processor)