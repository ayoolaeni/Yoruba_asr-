"""Data collator for CTC fine-tuning (identical to the original baseline notebook)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any  # transformers.Wav2Vec2Processor

    def __call__(self, features):
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, padding=True, return_tensors="pt")
        lab = self.processor.tokenizer.pad(labels, padding=True, return_tensors="pt")
        batch["labels"] = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        return batch
