"""
Retraining Agent
=================
Role: Adaptor
Perception: Labeled sample set S* from the oracle; clean training data;
            current model parameters theta.
Action: Incremental fine-tuning with gradient regularization.
Goal: Improve robust accuracy while preserving clean accuracy.

The Retraining Agent receives selected samples from the Selection Agent
(via the human oracle) and performs incremental model updates to improve
robustness. Key design choices:
- Incremental fine-tuning (not full retraining)
- Gradient regularization to prevent catastrophic forgetting
- Mixed training with clean + adversarial samples
"""

import torch
import torch.nn as nn
import copy
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..models.classifier import CTIClassifier
from .selection_agent import ScoredCandidate


class RetrainingAgent:
    """
    Retraining Agent: Performs incremental adversarial fine-tuning on
    selected high-uncertainty samples.

    Implements the retraining loss from Eq. 9 of the paper:
        L_total = L_CE(S*) + lambda * L_reg(theta)

    where L_CE is cross-entropy on selected samples and L_reg constrains
    parameter deviation for stability.
    """

    def __init__(
        self,
        classifier: CTIClassifier,
        learning_rate: float = 2e-5,
        epochs_per_iteration: int = 2,
        regularization_lambda: float = 0.01,
        clean_mix_ratio: float = 0.3,
        max_grad_norm: float = 1.0,
        epsilon: float = 0.01,
        max_length: int = 512
    ):
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.epochs_per_iteration = epochs_per_iteration
        self.reg_lambda = regularization_lambda
        self.clean_mix_ratio = clean_mix_ratio
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.max_length = max_length
        self.device = classifier.device

        # Store reference parameters for regularization
        self.reference_params: Optional[Dict[str, torch.Tensor]] = None

        # Retraining history
        self.history: List[Dict] = []

    def save_reference_parameters(self):
        """Save current model parameters as reference for regularization."""
        self.reference_params = {
            name: param.clone().detach()
            for name, param in self.classifier.get_model().named_parameters()
            if param.requires_grad
        }

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute gradient regularization loss L_reg(theta).

        L_reg = sum_i || theta_i - theta_i^ref ||^2

        This constrains parameter changes to prevent drastic shifts that
        could degrade clean accuracy.
        """
        if self.reference_params is None:
            return torch.tensor(0.0, device=self.device)

        reg_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.classifier.get_model().named_parameters():
            if param.requires_grad and name in self.reference_params:
                reg_loss += torch.sum(
                    (param - self.reference_params[name]) ** 2
                )
        return reg_loss

    def prepare_selected_data(
        self,
        selected_samples: List[ScoredCandidate]
    ) -> Tuple[List[str], List[int]]:
        """
        Prepare selected samples for retraining.

        In the paper's framework, the human oracle (security analyst)
        provides ground-truth labels for selected samples. Here we simulate
        the oracle using the true labels attached to each candidate.
        """
        texts = []
        labels = []

        for scored in selected_samples:
            candidate = scored.candidate
            # Oracle provides the true label
            texts.append(candidate.text)
            labels.append(candidate.true_label)

        return texts, labels

    def create_mixed_dataloader(
        self,
        adv_texts: List[str],
        adv_labels: List[int],
        clean_loader: Optional[DataLoader] = None,
        batch_size: int = 16
    ) -> DataLoader:
        """
        Create a mixed DataLoader combining adversarial and clean samples.

        This implements the "Mixed Training" strategy from Section 4.5:
        selected adversarial samples are combined with a subset of clean
        training data during fine-tuning.
        """
        tokenizer = self.classifier.get_tokenizer()

        # Tokenize adversarial samples
        adv_encodings = tokenizer(
            list(adv_texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        all_input_ids = [adv_encodings["input_ids"]]
        all_attention_masks = [adv_encodings["attention_mask"]]
        all_labels = [torch.tensor(adv_labels, dtype=torch.long)]

        # Mix in clean samples directly from their tokenized tensors (no decode round-trip)
        if clean_loader is not None and self.clean_mix_ratio > 0:
            n_clean_to_add = int(len(adv_texts) * self.clean_mix_ratio)
            clean_added = 0
            clean_ids_list, clean_mask_list, clean_label_list = [], [], []

            for batch in clean_loader:
                if clean_added >= n_clean_to_add:
                    break
                batch_ids = batch["input_ids"]
                batch_mask = batch["attention_mask"]
                batch_labels = batch["labels"]
                remaining = n_clean_to_add - clean_added
                take = min(len(batch_labels), remaining)
                clean_ids_list.append(batch_ids[:take])
                clean_mask_list.append(batch_mask[:take])
                clean_label_list.append(batch_labels[:take])
                clean_added += take

            if clean_ids_list:
                # Pad/truncate clean tensors to match adversarial max_length
                clean_ids = torch.cat(clean_ids_list, dim=0)
                clean_mask = torch.cat(clean_mask_list, dim=0)
                clean_labels = torch.cat(clean_label_list, dim=0)
                # Handle length mismatch between loaders
                adv_len = adv_encodings["input_ids"].shape[1]
                clean_len = clean_ids.shape[1]
                if clean_len > adv_len:
                    clean_ids = clean_ids[:, :adv_len]
                    clean_mask = clean_mask[:, :adv_len]
                elif clean_len < adv_len:
                    pad = torch.zeros(clean_ids.shape[0], adv_len - clean_len, dtype=clean_ids.dtype)
                    clean_ids = torch.cat([clean_ids, pad], dim=1)
                    clean_mask = torch.cat([clean_mask, pad], dim=1)
                all_input_ids.append(clean_ids)
                all_attention_masks.append(clean_mask)
                all_labels.append(clean_labels)

        dataset = TensorDataset(
            torch.cat(all_input_ids, dim=0),
            torch.cat(all_attention_masks, dim=0),
            torch.cat(all_labels, dim=0),
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def fgsm_perturbation(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        """Generate FGSM adversarial embeddings for adversarial retraining."""
        embeddings_clone = embeddings.clone().detach().requires_grad_(True)

        outputs = self.classifier.get_model()(
            inputs_embeds=embeddings_clone,
            attention_mask=attention_mask,
            input_ids=None
        )
        loss = criterion(outputs.logits, labels)
        loss.backward()

        grad_sign = embeddings_clone.grad.sign()
        adv_embeddings = embeddings_clone + self.epsilon * grad_sign
        return adv_embeddings.detach()

    def retrain_iteration(
        self,
        selected_samples: List[ScoredCandidate],
        clean_loader: Optional[DataLoader] = None,
        dev_loader: Optional[DataLoader] = None,
        class_weights: Optional[torch.Tensor] = None,
        iteration: int = 0
    ) -> Dict:
        r"""
        Perform one active learning retraining iteration.

        This implements Steps 4-5 of Algorithm 1:
        Step 4: theta <- FineTune(theta, L, epochs=2)
        Step 5: U <- U \ S_t

        Args:
            selected_samples: Selected high-uncertainty samples (from SelectionAgent)
            clean_loader: Clean training data for mixed training
            dev_loader: Validation loader for evaluation
            class_weights: Class weights for loss function
            iteration: Current AL iteration number

        Returns:
            Dictionary of retraining metrics
        """
        model = self.classifier.get_model()

        # Save reference parameters for regularization
        self.save_reference_parameters()

        # Prepare data
        adv_texts, adv_labels = self.prepare_selected_data(selected_samples)

        if not adv_texts:
            return {"status": "no_samples", "iteration": iteration}

        # Create mixed dataloader
        mixed_loader = self.create_mixed_dataloader(
            adv_texts, adv_labels,
            clean_loader=clean_loader,
            batch_size=16
        )

        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        # Setup loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Fine-tune for a few epochs
        iteration_losses = []
        model.train()

        for epoch in range(self.epochs_per_iteration):
            epoch_loss = 0.0
            n_batches = 0

            for batch in mixed_loader:
                optimizer.zero_grad()

                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                # Get embeddings for adversarial training
                embeddings = self.classifier.get_embeddings(input_ids)

                # Clean forward pass
                clean_outputs = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    input_ids=None
                )
                clean_loss = criterion(clean_outputs.logits, labels)

                # FGSM adversarial forward pass
                adv_embeddings = self.fgsm_perturbation(
                    embeddings, attention_mask, labels, criterion
                )
                adv_outputs = model(
                    inputs_embeds=adv_embeddings,
                    attention_mask=attention_mask,
                    input_ids=None
                )
                adv_loss = criterion(adv_outputs.logits, labels)

                # Combined loss: L_total = L_CE(S*) + lambda * L_reg(theta)
                ce_loss = 0.7 * clean_loss + 0.3 * adv_loss
                reg_loss = self.compute_regularization_loss()
                total_loss = ce_loss + self.reg_lambda * reg_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            iteration_losses.append(avg_epoch_loss)

        # Evaluate if dev_loader provided
        dev_acc, dev_f1 = 0.0, 0.0
        if dev_loader is not None:
            dev_acc, dev_f1 = self._evaluate(dev_loader)

        result = {
            "iteration": iteration,
            "n_samples": len(adv_texts),
            "epochs": self.epochs_per_iteration,
            "avg_loss": float(np.mean(iteration_losses)),
            "final_loss": float(iteration_losses[-1]) if iteration_losses else 0.0,
            "dev_accuracy": dev_acc,
            "dev_f1": dev_f1,
        }

        self.history.append(result)

        print(f"Retraining Agent [Iter {iteration}]: "
              f"Fine-tuned on {len(adv_texts)} samples for {self.epochs_per_iteration} epochs, "
              f"loss={result['avg_loss']:.4f}, dev_acc={dev_acc:.4f}, dev_f1={dev_f1:.4f}")

        return result

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a DataLoader."""
        model = self.classifier.get_model()
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)

        return accuracy, f1

    def get_history(self) -> List[Dict]:
        """Return retraining history."""
        return self.history.copy()
