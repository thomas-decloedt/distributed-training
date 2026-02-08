"""LoRA dataloader — strict typing, Pydantic models only."""

from __future__ import annotations

from collections.abc import Sequence

from common.dataloader import create_distributed_dataloader
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from lora.config import TrainingConfig
from lora.models import TokenizedBatch, TokenizedExample


class TextDataset(Dataset[TokenizedExample]):
    """Dataset of tokenized text examples."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        dataset_config: str,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        raw = load_dataset(dataset_name, dataset_config, split="train")
        self.examples: list[TokenizedExample] = []
        for item in raw:
            text = item.get("text", "")
            if not text.strip():
                continue
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )
            labels = list(enc["input_ids"])
            for i in range(len(labels)):
                if enc["attention_mask"][i] == 0:
                    labels[i] = -100
            self.examples.append(
                TokenizedExample(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=labels,
                )
            )
            if len(self.examples) >= 1000:
                break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedExample:
        return self.examples[idx]


def collate_fn(batch: Sequence[TokenizedExample]) -> TokenizedBatch:
    """Collate TokenizedExamples into a TokenizedBatch of tensors."""
    import torch

    input_ids = [b.input_ids for b in batch]
    attention_mask = [b.attention_mask for b in batch]
    labels = [b.labels for b in batch]
    return TokenizedBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def get_dataloader(
    config: TrainingConfig,
    tokenizer: AutoTokenizer,
    rank: int,
    world_size: int,
) -> DataLoader[TokenizedBatch]:
    """Create distributed dataloader for LoRA training."""
    dataset = TextDataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        max_length=config.max_length,
    )
    return create_distributed_dataloader(
        dataset,
        batch_size=config.batch_size,
        rank=rank,
        world_size=world_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
