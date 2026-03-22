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
from itertools import cycle
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
        learning_rate: float = 5e-6,
        epochs_per_iteration: int = 3,
        regularization_lambda: float = 0.1,
        clean_mix_ratio: float = 3.0,
        max_grad_norm: float = 1.0,
        epsilon: float = 0.05,
        max_length: int = 512,
        # Enhancement parameters
        p_fgsm_clean: float = 0.3,       # Prob of applying FGSM to clean batches
        total_iterations: int = 5,        # For progressive regularization
        epsilon_schedule: Optional[List[float]] = None,  # Epsilon curriculum
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
        self.p_fgsm_clean = p_fgsm_clean
        self.total_iterations = total_iterations
        self.epsilon_schedule = epsilon_schedule or [0.02, 0.04, 0.06, 0.08, 0.10]

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

        # Mix in clean samples — use ratio relative to adversarial count
        # clean_mix_ratio=2.0 means 2x more clean samples than adversarial
        if clean_loader is not None and self.clean_mix_ratio > 0:
            n_clean_to_add = max(int(len(adv_texts) * self.clean_mix_ratio), len(adv_texts))
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
        position_ids = self.classifier._get_position_ids(
            embeddings.shape[1], embeddings.shape[0]
        )

        outputs = self.classifier.get_model()(
            inputs_embeds=embeddings_clone,
            attention_mask=attention_mask,
            position_ids=position_ids
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

        Trains on the FULL clean training set with FGSM adversarial augmentation,
        plus the selected adversarial samples as additional augmentation.
        This ensures the model retains clean accuracy while learning robustness.

        Args:
            selected_samples: Selected high-utility adversarial samples
            clean_loader: FULL clean training DataLoader (used entirely)
            dev_loader: Validation loader for evaluation
            class_weights: Class weights for loss function
            iteration: Current AL iteration number

        Returns:
            Dictionary of retraining metrics
        """
        model = self.classifier.get_model()

        # Save reference parameters for regularization
        self.save_reference_parameters()

        # Prepare selected adversarial texts as a supplementary loader
        adv_texts, adv_labels = self.prepare_selected_data(selected_samples)

        if not adv_texts:
            return {"status": "no_samples", "iteration": iteration}

        # Create adversarial supplement loader (just the selected samples)
        adv_loader = None
        if adv_texts:
            tokenizer = self.classifier.get_tokenizer()
            adv_encodings = tokenizer(
                adv_texts, truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt"
            )
            adv_dataset = TensorDataset(
                adv_encodings["input_ids"],
                adv_encodings["attention_mask"],
                torch.tensor(adv_labels, dtype=torch.long),
            )
            adv_loader = DataLoader(adv_dataset, batch_size=16, shuffle=True)

        # Setup optimizer — lower LR for fine-tuning
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        # Setup loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # ================================================================
        # INTERLEAVED ADVERSARIAL TRAINING (6 enhancements)
        # ================================================================
        # 1. Interleaved: every clean batch also gets an adversarial step
        # 2. FGSM on random clean batches (p=0.3, no annotation needed)
        # 3. Multiple epochs (3 per iteration)
        # 4. Higher LR (5e-6)
        # 5. Progressive regularization (weak early, strong late)
        # 6. Epsilon curriculum across iterations
        # ================================================================

        # Enhancement 5: Progressive regularization
        effective_lambda = self.reg_lambda * max(0.1, iteration / max(self.total_iterations, 1))

        # Enhancement 6: Epsilon curriculum
        if 0 < iteration <= len(self.epsilon_schedule):
            current_epsilon = self.epsilon_schedule[iteration - 1]
        else:
            current_epsilon = self.epsilon
        old_epsilon = self.epsilon
        self.epsilon = current_epsilon

        iteration_losses = []
        model.train()

        # Enhancement 1: Cycling iterator over selected adversarial samples
        adv_cycle = cycle(adv_loader) if adv_loader is not None else None

        for epoch in range(self.epochs_per_iteration):
            epoch_loss = 0.0
            n_batches = 0
            n_adv_updates = 0

            if clean_loader is None:
                continue

            for batch in clean_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # --- Clean loss on full training batch ---
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_loss = criterion(outputs.logits, labels)

                # --- Enhancement 2: FGSM on random clean batches (no annotation) ---
                adv_clean_loss = torch.tensor(0.0, device=self.device)
                if np.random.random() < self.p_fgsm_clean:
                    embeddings = self.classifier.get_embeddings(input_ids)
                    position_ids = self.classifier._get_position_ids(
                        input_ids.shape[1], input_ids.shape[0]
                    )
                    adv_emb = self.fgsm_perturbation(
                        embeddings, attention_mask, labels, criterion
                    )
                    adv_out = model(
                        inputs_embeds=adv_emb,
                        attention_mask=attention_mask,
                        position_ids=position_ids
                    )
                    adv_clean_loss = criterion(adv_out.logits, labels)
                    n_adv_updates += 1

                # --- Enhancement 1: Interleaved adversarial from selected pool ---
                adv_selected_loss = torch.tensor(0.0, device=self.device)
                if adv_cycle is not None:
                    adv_batch = next(adv_cycle)
                    adv_ids = adv_batch[0].to(self.device)
                    adv_mask = adv_batch[1].to(self.device)
                    adv_lbls = adv_batch[2].to(self.device)

                    adv_emb = self.classifier.get_embeddings(adv_ids)
                    position_ids = self.classifier._get_position_ids(
                        adv_ids.shape[1], adv_ids.shape[0]
                    )
                    adv_perturbed = self.fgsm_perturbation(
                        adv_emb, adv_mask, adv_lbls, criterion
                    )
                    adv_out = model(
                        inputs_embeds=adv_perturbed,
                        attention_mask=adv_mask,
                        position_ids=position_ids
                    )
                    adv_selected_loss = criterion(adv_out.logits, adv_lbls)
                    n_adv_updates += 1

                # --- Combined loss ---
                total_loss = (
                    0.6 * clean_loss
                    + 0.2 * adv_clean_loss
                    + 0.2 * adv_selected_loss
                    + effective_lambda * self.compute_regularization_loss()
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            iteration_losses.append(avg_epoch_loss)
            print(f"    Epoch {epoch+1}/{self.epochs_per_iteration}: "
                  f"loss={avg_epoch_loss:.4f}, adv_updates={n_adv_updates}, "
                  f"eps={current_epsilon:.3f}, lambda={effective_lambda:.3f}")

        self.epsilon = old_epsilon

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
