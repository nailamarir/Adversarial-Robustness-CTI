"""
Model Module
CTI Text Classifier based on DistilBERT/BERT variants
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Tuple, Dict, Optional
import os


class CTIClassifier:
    """Wrapper class for CTI text classification model"""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 16,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizer = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        print(f"Number of labels: {self.num_labels}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # DistilBERT uses 'seq_classif_dropout' / 'dropout'; BERT/RoBERTa use 'hidden_dropout_prob'
        dropout_kwargs = {}
        if "distilbert" in self.model_name.lower():
            dropout_kwargs["seq_classif_dropout"] = 0.2
        else:
            dropout_kwargs["hidden_dropout_prob"] = 0.2
            dropout_kwargs["attention_probs_dropout_prob"] = 0.2

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            **dropout_kwargs,
        )

        self.model.to(self.device)
        print(f"Model loaded successfully!")

    def get_model(self) -> PreTrainedModel:
        """Get the underlying model"""
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer"""
        return self.tokenizer

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get word embeddings from input IDs"""
        # Check distilbert/roberta before bert — SecRoBERTa has 'roberta', SecBERT has 'bert'
        if hasattr(self.model, 'distilbert'):
            return self.model.distilbert.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'roberta'):
            return self.model.roberta.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'bert'):
            return self.model.bert.embeddings.word_embeddings(input_ids)
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")

    def _get_position_ids(self, seq_length: int, batch_size: int = 1) -> torch.Tensor:
        """Generate position IDs for inputs_embeds forward pass."""
        return torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1)

    def forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass using embeddings instead of input IDs"""
        position_ids = self._get_position_ids(embeddings.shape[1], embeddings.shape[0])
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        return outputs

    def predict(
        self,
        text: str,
        max_length: int = 512
    ) -> int:
        """Predict class for a single text"""
        self.model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)

        return prediction.item()

    def predict_batch(
        self,
        texts: list,
        max_length: int = 512,
        batch_size: int = 32
    ) -> list:
        """Predict classes for a batch of texts"""
        self.model.eval()
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_preds.cpu().tolist())

        return predictions

    def get_probabilities(
        self,
        text: str,
        max_length: int = 512
    ) -> torch.Tensor:
        """Get class probabilities for a single text"""
        self.model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.squeeze()

    def train_mode(self) -> None:
        """Set model to training mode"""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode"""
        self.model.eval()

    def save(self, path: str) -> None:
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from: {path}")

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }


def load_model(
    path: str,
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 16,
    device: Optional[str] = None
) -> CTIClassifier:
    """Load a saved model"""
    classifier = CTIClassifier(
        model_name=model_name,
        num_labels=num_labels,
        device=device
    )
    classifier.load(path)
    return classifier


def save_model(classifier: CTIClassifier, path: str) -> None:
    """Save model weights"""
    classifier.save(path)
