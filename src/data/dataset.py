"""
Dataset and DataLoader Module
PyTorch Dataset class and DataLoader utilities for CTI data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple, Optional
import pandas as pd


class CTIDataset(Dataset):
    """PyTorch Dataset for CTI text classification"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, dev, and test sets"""

    train_dataset = CTIDataset(
        texts=train_df["full_text"].tolist(),
        labels=train_df["label_id_encoded"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    dev_dataset = CTIDataset(
        texts=dev_df["full_text"].tolist(),
        labels=dev_df["label_id_encoded"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = CTIDataset(
        texts=test_df["full_text"].tolist(),
        labels=test_df["label_id_encoded"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Created DataLoaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Dev:   {len(dev_loader)} batches ({len(dev_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, dev_loader, test_loader


class AdversarialDataset(Dataset):
    """Dataset for adversarial examples evaluation"""

    def __init__(
        self,
        original_texts: List[str],
        adversarial_texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.original_texts = original_texts
        self.adversarial_texts = adversarial_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.original_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        orig_text = str(self.original_texts[idx])
        adv_text = str(self.adversarial_texts[idx])
        label = int(self.labels[idx])

        orig_encoding = self.tokenizer(
            orig_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        adv_encoding = self.tokenizer(
            adv_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "orig_input_ids": orig_encoding["input_ids"].squeeze(0),
            "orig_attention_mask": orig_encoding["attention_mask"].squeeze(0),
            "adv_input_ids": adv_encoding["input_ids"].squeeze(0),
            "adv_attention_mask": adv_encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
