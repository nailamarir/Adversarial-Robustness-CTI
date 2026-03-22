"""
Training Module
Baseline and Adversarial trainers for CTI classification
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..models.classifier import CTIClassifier
from .losses import compute_class_weights, WeightedCrossEntropyLoss, TRADESLoss


class BaselineTrainer:
    """Trainer for baseline (non-adversarial) model"""

    def __init__(
        self,
        classifier: CTIClassifier,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        patience: int = 3
    ):
        self.classifier = classifier
        self.model = classifier.get_model()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.device = classifier.device

        # Setup optimizer with weight decay to reduce overfitting
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        total_steps = len(train_loader) * num_epochs
        warmup = max(warmup_steps, int(0.1 * total_steps))  # at least 10% warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps
        )

        # Setup loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            "train_loss": [],
            "dev_acc": [],
            "dev_f1": [],
            "best_f1": 0.0
        }

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="macro")

        return accuracy, f1

    def train(self) -> Dict:
        """Full training loop with best-checkpoint restoration."""
        import copy
        print("=" * 60)
        print("Starting Baseline Training")
        print("=" * 60)

        best_f1 = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Evaluate
            dev_acc, dev_f1 = self.evaluate(self.dev_loader)
            self.history["dev_acc"].append(dev_acc)
            self.history["dev_f1"].append(dev_f1)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Dev Acc: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")

            # Save best checkpoint
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                self.history["best_f1"] = best_f1
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        # Restore best checkpoint before returning
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best checkpoint (F1={best_f1:.4f})")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Dev F1: {best_f1:.4f}")
        print("=" * 60)

        return self.history


class AdversarialTrainer:
    """Trainer for adversarial (FGSM) training"""

    def __init__(
        self,
        classifier: CTIClassifier,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        num_epochs: int = 1,
        learning_rate: float = 1e-5,
        epsilon: float = 0.005,
        clean_weight: float = 0.7,
        adv_weight: float = 0.3,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.classifier = classifier
        self.model = classifier.get_model()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.clean_weight = clean_weight
        self.adv_weight = adv_weight
        self.device = classifier.device

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Setup loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            "train_loss": [],
            "clean_loss": [],
            "adv_loss": [],
            "dev_acc": [],
            "dev_f1": []
        }

    def fgsm_attack(
        self,
        embeddings: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Generate FGSM adversarial embeddings"""
        grad = embeddings.grad
        adv_embeddings = embeddings + epsilon * grad.sign()
        return adv_embeddings

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch with FGSM"""
        self.model.train()
        total_loss = 0.0
        total_clean_loss = 0.0
        total_adv_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Adversarial Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Get embeddings
            embeddings = self.classifier.get_embeddings(input_ids)
            embeddings_for_grad = embeddings.clone().detach().requires_grad_(True)

            # Generate position_ids for inputs_embeds forward passes
            position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).expand(input_ids.shape[0], -1)

            # Forward pass on detached embeddings to get FGSM gradient
            grad_outputs = self.model(
                inputs_embeds=embeddings_for_grad,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            grad_loss = self.criterion(grad_outputs.logits, labels)
            grad_loss.backward()

            # Generate adversarial embeddings using gradient sign
            grad_sign = embeddings_for_grad.grad.sign()
            adv_embeddings = (embeddings + self.epsilon * grad_sign).detach().requires_grad_(True)

            # Now compute clean and adversarial losses together for proper gradient flow
            clean_outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            clean_loss = self.criterion(clean_outputs.logits, labels)

            adv_outputs = self.model(
                inputs_embeds=adv_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            adv_loss = self.criterion(adv_outputs.logits, labels)

            # Combined loss — single backward pass, clean gradient flow
            combined_loss = (
                self.clean_weight * clean_loss +
                self.adv_weight * adv_loss
            )

            self.optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += combined_loss.item()
            total_clean_loss += clean_loss.item()
            total_adv_loss += adv_loss.item()

            progress_bar.set_postfix({
                "loss": f"{combined_loss.item():.4f}",
                "clean": f"{clean_loss.item():.4f}",
                "adv": f"{adv_loss.item():.4f}"
            })

        n_batches = len(self.train_loader)
        return (
            total_loss / n_batches,
            total_clean_loss / n_batches,
            total_adv_loss / n_batches
        )

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="macro")

        return accuracy, f1

    def train(self) -> Dict:
        """Full adversarial training loop"""
        print("=" * 60)
        print("Starting Adversarial (FGSM) Training")
        print(f"Epsilon: {self.epsilon}")
        print(f"Loss weights: Clean={self.clean_weight}, Adv={self.adv_weight}")
        print("=" * 60)

        for epoch in range(self.num_epochs):
            # Train
            train_loss, clean_loss, adv_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["clean_loss"].append(clean_loss)
            self.history["adv_loss"].append(adv_loss)

            # Evaluate
            dev_acc, dev_f1 = self.evaluate(self.dev_loader)
            self.history["dev_acc"].append(dev_acc)
            self.history["dev_f1"].append(dev_f1)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Clean: {clean_loss:.4f}, Adv: {adv_loss:.4f})")
            print(f"  Dev Acc: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")

        print("\n" + "=" * 60)
        print("Adversarial Training Complete!")
        print("=" * 60)

        return self.history


class PGDTrainer:
    """
    Trainer for PGD (Projected Gradient Descent) adversarial training

    PGD is a stronger multi-step attack compared to single-step FGSM.
    It iteratively applies small perturbations and projects back to epsilon ball.
    """

    def __init__(
        self,
        classifier: CTIClassifier,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        epsilon: float = 0.01,
        alpha: float = 0.003,
        num_steps: int = 7,
        clean_weight: float = 0.5,
        adv_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.classifier = classifier
        self.model = classifier.get_model()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.clean_weight = clean_weight
        self.adv_weight = adv_weight
        self.device = classifier.device

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Setup loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            "train_loss": [],
            "clean_loss": [],
            "adv_loss": [],
            "dev_acc": [],
            "dev_f1": []
        }

    def pgd_attack(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate PGD adversarial embeddings

        Multi-step attack with projection to epsilon ball
        """
        # Start with small random perturbation
        delta = torch.zeros_like(embeddings).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for step in range(self.num_steps):
            # Forward pass with perturbed embeddings
            adv_embeddings = embeddings + delta

            outputs = self.model(
                inputs_embeds=adv_embeddings,
                attention_mask=attention_mask,
                input_ids=None
            )

            loss = self.criterion(outputs.logits, labels)

            # Backward pass
            loss.backward(retain_graph=True)

            # Update delta
            grad = delta.grad.detach()
            delta = delta + self.alpha * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = delta.detach().requires_grad_(True)

        return (embeddings + delta).detach()

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch with PGD"""
        self.model.train()
        total_loss = 0.0
        total_clean_loss = 0.0
        total_adv_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="PGD Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Get embeddings
            embeddings = self.classifier.get_embeddings(input_ids)

            # Clean forward pass
            clean_outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                input_ids=None
            )
            clean_loss = self.criterion(clean_outputs.logits, labels)

            # Generate PGD adversarial embeddings
            adv_embeddings = self.pgd_attack(embeddings.detach(), attention_mask, labels)

            # Adversarial forward pass
            adv_outputs = self.model(
                inputs_embeds=adv_embeddings,
                attention_mask=attention_mask,
                input_ids=None
            )
            adv_loss = self.criterion(adv_outputs.logits, labels)

            # Combined loss
            combined_loss = (
                self.clean_weight * clean_loss +
                self.adv_weight * adv_loss
            )

            # Update weights
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += combined_loss.item()
            total_clean_loss += clean_loss.item()
            total_adv_loss += adv_loss.item()

            progress_bar.set_postfix({
                "loss": f"{combined_loss.item():.4f}",
                "clean": f"{clean_loss.item():.4f}",
                "adv": f"{adv_loss.item():.4f}"
            })

        n_batches = len(self.train_loader)
        return (
            total_loss / n_batches,
            total_clean_loss / n_batches,
            total_adv_loss / n_batches
        )

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="macro")

        return accuracy, f1

    def train(self) -> Dict:
        """Full PGD training loop"""
        print("=" * 60)
        print("Starting PGD Adversarial Training")
        print(f"Epsilon: {self.epsilon}, Alpha: {self.alpha}, Steps: {self.num_steps}")
        print(f"Loss weights: Clean={self.clean_weight}, Adv={self.adv_weight}")
        print("=" * 60)

        for epoch in range(self.num_epochs):
            train_loss, clean_loss, adv_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["clean_loss"].append(clean_loss)
            self.history["adv_loss"].append(adv_loss)

            dev_acc, dev_f1 = self.evaluate(self.dev_loader)
            self.history["dev_acc"].append(dev_acc)
            self.history["dev_f1"].append(dev_f1)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Clean: {clean_loss:.4f}, Adv: {adv_loss:.4f})")
            print(f"  Dev Acc: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")

        print("\n" + "=" * 60)
        print("PGD Training Complete!")
        print("=" * 60)

        return self.history


class TRADESTrainer:
    """
    Trainer using TRADES loss

    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
    provides a principled way to balance accuracy and robustness.

    Loss = CE(clean, labels) + beta * KL(adv || clean)
    """

    def __init__(
        self,
        classifier: CTIClassifier,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        epsilon: float = 0.01,
        step_size: float = 0.003,
        num_steps: int = 7,
        beta: float = 6.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.classifier = classifier
        self.model = classifier.get_model()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.beta = beta
        self.device = classifier.device

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Setup TRADES loss
        self.criterion = TRADESLoss(beta=beta, class_weights=class_weights)

        # Standard CE for generating adversarial examples
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Training history
        self.history = {
            "train_loss": [],
            "natural_loss": [],
            "robust_loss": [],
            "dev_acc": [],
            "dev_f1": []
        }

    def trades_pgd_attack(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        clean_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples for TRADES using PGD

        Unlike standard PGD, TRADES maximizes KL divergence from clean predictions
        """
        # Start with small random perturbation
        delta = torch.zeros_like(embeddings).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for step in range(self.num_steps):
            adv_embeddings = embeddings + delta

            adv_outputs = self.model(
                inputs_embeds=adv_embeddings,
                attention_mask=attention_mask,
                input_ids=None
            )

            # Maximize KL divergence from clean predictions
            clean_probs = nn.functional.softmax(clean_logits.detach(), dim=1)
            adv_log_probs = nn.functional.log_softmax(adv_outputs.logits, dim=1)

            loss = self.kl_loss(adv_log_probs, clean_probs)
            loss.backward(retain_graph=True)

            # Update delta
            grad = delta.grad.detach()
            delta = delta + self.step_size * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = delta.detach().requires_grad_(True)

        return (embeddings + delta).detach()

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch with TRADES"""
        self.model.train()
        total_loss = 0.0
        total_natural_loss = 0.0
        total_robust_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="TRADES Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Get embeddings
            embeddings = self.classifier.get_embeddings(input_ids)

            # Clean forward pass
            clean_outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                input_ids=None
            )

            # Generate TRADES adversarial embeddings
            adv_embeddings = self.trades_pgd_attack(
                embeddings.detach(),
                attention_mask,
                clean_outputs.logits
            )

            # Adversarial forward pass
            adv_outputs = self.model(
                inputs_embeds=adv_embeddings,
                attention_mask=attention_mask,
                input_ids=None
            )

            # Compute TRADES loss
            total, natural, robust = self.criterion(
                clean_outputs.logits,
                adv_outputs.logits,
                labels
            )

            # Update weights
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += total.item()
            total_natural_loss += natural.item()
            total_robust_loss += robust.item()

            progress_bar.set_postfix({
                "loss": f"{total.item():.4f}",
                "nat": f"{natural.item():.4f}",
                "rob": f"{robust.item():.4f}"
            })

        n_batches = len(self.train_loader)
        return (
            total_loss / n_batches,
            total_natural_loss / n_batches,
            total_robust_loss / n_batches
        )

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="macro")

        return accuracy, f1

    def train(self) -> Dict:
        """Full TRADES training loop"""
        print("=" * 60)
        print("Starting TRADES Training")
        print(f"Beta: {self.beta} (higher = more robust)")
        print(f"Epsilon: {self.epsilon}, Step size: {self.step_size}, Steps: {self.num_steps}")
        print("=" * 60)

        for epoch in range(self.num_epochs):
            train_loss, natural_loss, robust_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["natural_loss"].append(natural_loss)
            self.history["robust_loss"].append(robust_loss)

            dev_acc, dev_f1 = self.evaluate(self.dev_loader)
            self.history["dev_acc"].append(dev_acc)
            self.history["dev_f1"].append(dev_f1)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Natural: {natural_loss:.4f}, Robust: {robust_loss:.4f})")
            print(f"  Dev Acc: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")

        print("\n" + "=" * 60)
        print("TRADES Training Complete!")
        print("=" * 60)

        return self.history
