import torch
import librosa
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union

SAMPLE_RATE = 16000


# ── Dataset ──────────────────────────────────────────────────────────────────

class AzerbaijaniASRDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.data      = list(hf_dataset)   # convert to plain list — no HF overhead
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample    = self.data[idx]
        audio, _  = librosa.load(sample["path"], sr=SAMPLE_RATE)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE
        ).input_features[0]

        labels = self.processor.tokenizer(sample["sentence"]).input_ids

        return {
            "input_features": input_features,
            "labels":         labels,
        }


def build_dataset(hf_dataset, processor, num_samples: int = None):
    if num_samples is not None:
        hf_dataset = hf_dataset.select(range(min(num_samples, len(hf_dataset))))
    return AzerbaijaniASRDataset(hf_dataset, processor)


# ── Data Collator ─────────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut BOS token
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_collator(processor) -> WhisperDataCollator:
    return WhisperDataCollator(processor=processor)