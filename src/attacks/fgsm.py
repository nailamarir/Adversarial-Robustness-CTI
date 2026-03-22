"""
FGSM (Fast Gradient Sign Method) Attack Module
Embedding-level adversarial attack for evaluation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


class FGSMAttack:
    """
    Fast Gradient Sign Method attack for text classification models
    Operates at the embedding level
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        epsilon: float = 0.1,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get word embeddings from model"""
        # Check distilbert/roberta before bert — SecBERT has 'bert', SecRoBERTa has 'roberta'
        if hasattr(self.model, 'distilbert'):
            return self.model.distilbert.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'roberta'):
            return self.model.roberta.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'bert'):
            return self.model.bert.embeddings.word_embeddings(input_ids)
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate position IDs for inputs_embeds forward pass."""
        seq_length = input_ids.shape[1]
        return torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

    def generate_adversarial_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial embeddings using FGSM

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: True labels

        Returns:
            Adversarial embeddings
        """
        self.model.eval()

        # Get embeddings and enable gradients
        embeddings = self.get_embeddings(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)
        position_ids = self._get_position_ids(input_ids)

        # Forward pass
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        # Compute loss
        loss = self.criterion(outputs.logits, labels)

        # Backward pass to get gradients
        loss.backward()

        # Generate adversarial embeddings
        grad_sign = embeddings.grad.sign()
        adv_embeddings = embeddings + self.epsilon * grad_sign

        return adv_embeddings.detach()

    def attack_text(
        self,
        text: str,
        true_label: int,
        max_length: int = 512
    ) -> Tuple[int, int, bool]:
        """
        Attack a single text sample

        Args:
            text: Input text
            true_label: True label
            max_length: Maximum sequence length

        Returns:
            Tuple of (original_pred, adv_pred, attack_success)
        """
        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = torch.tensor([true_label]).to(self.device)

        # Original prediction
        with torch.no_grad():
            orig_outputs = self.model(**inputs)
            orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()

        # Generate adversarial embeddings
        adv_embeddings = self.generate_adversarial_embeddings(
            input_ids, attention_mask, labels
        )

        # Adversarial prediction
        position_ids = self._get_position_ids(input_ids)
        with torch.no_grad():
            adv_outputs = self.model(
                inputs_embeds=adv_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            adv_pred = torch.argmax(adv_outputs.logits, dim=-1).item()

        # Attack success: originally correct, now wrong
        attack_success = (orig_pred == true_label) and (adv_pred != true_label)

        return orig_pred, adv_pred, attack_success

    def attack_batch(
        self,
        texts: list,
        labels: list,
        max_length: int = 512
    ) -> dict:
        """
        Attack a batch of texts

        Args:
            texts: List of input texts
            labels: List of true labels
            max_length: Maximum sequence length

        Returns:
            Dictionary with attack results
        """
        results = {
            "original_preds": [],
            "adv_preds": [],
            "attack_success": [],
            "originally_correct": 0,
            "successful_attacks": 0
        }

        for text, label in zip(texts, labels):
            orig_pred, adv_pred, success = self.attack_text(
                text, label, max_length
            )

            results["original_preds"].append(orig_pred)
            results["adv_preds"].append(adv_pred)
            results["attack_success"].append(success)

            if orig_pred == label:
                results["originally_correct"] += 1
            if success:
                results["successful_attacks"] += 1

        # Compute ASR
        if results["originally_correct"] > 0:
            results["asr"] = (
                results["successful_attacks"] / results["originally_correct"]
            )
        else:
            results["asr"] = 0.0

        return results


class PGDAttack:
    """
    Projected Gradient Descent attack (stronger than FGSM)
    Multi-step attack with projection
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 10,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.alpha = alpha  # Step size
        self.num_steps = num_steps
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get word embeddings from model"""
        if hasattr(self.model, 'distilbert'):
            return self.model.distilbert.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'roberta'):
            return self.model.roberta.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'bert'):
            return self.model.bert.embeddings.word_embeddings(input_ids)
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate position IDs for inputs_embeds forward pass."""
        seq_length = input_ids.shape[1]
        return torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

    def generate_adversarial_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial embeddings using PGD"""
        self.model.eval()

        # Get original embeddings
        orig_embeddings = self.get_embeddings(input_ids).detach()
        position_ids = self._get_position_ids(input_ids)

        # Initialize perturbation
        delta = torch.zeros_like(orig_embeddings).uniform_(
            -self.epsilon, self.epsilon
        )
        delta.requires_grad = True

        for _ in range(self.num_steps):
            # Forward pass with perturbed embeddings
            adv_embeddings = orig_embeddings + delta

            outputs = self.model(
                inputs_embeds=adv_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids
            )

            loss = self.criterion(outputs.logits, labels)
            loss.backward()

            # Update perturbation
            grad_sign = delta.grad.sign()
            delta = delta + self.alpha * grad_sign

            # Project back to epsilon ball
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = delta.detach().requires_grad_(True)

        return (orig_embeddings + delta).detach()
