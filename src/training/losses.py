"""
Loss Functions Module
Custom loss functions for CTI classification
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, List


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute balanced class weights for imbalanced datasets

    Args:
        labels: Array of training labels
        num_classes: Total number of classes
        device: Device to place weights tensor

    Returns:
        Tensor of class weights
    """
    unique_classes = np.unique(labels)

    # Compute weights for present classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels
    )

    # Create full weight vector (some classes may be missing)
    weights = torch.ones(num_classes, dtype=torch.float)
    for i, cls in enumerate(unique_classes):
        weights[int(cls)] = class_weights[i]

    return weights.to(device)


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for imbalanced classification"""

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.weights = weights
        self.label_smoothing = label_smoothing

        if weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=label_smoothing
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses learning on hard examples
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AdversarialLoss(nn.Module):
    """
    Combined loss for adversarial training
    Combines clean and adversarial losses with configurable weights
    """

    def __init__(
        self,
        base_loss: nn.Module,
        clean_weight: float = 0.7,
        adv_weight: float = 0.3
    ):
        super().__init__()
        self.base_loss = base_loss
        self.clean_weight = clean_weight
        self.adv_weight = adv_weight

    def forward(
        self,
        clean_logits: torch.Tensor,
        adv_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        clean_loss = self.base_loss(clean_logits, targets)
        adv_loss = self.base_loss(adv_logits, targets)

        combined_loss = (
            self.clean_weight * clean_loss +
            self.adv_weight * adv_loss
        )

        return combined_loss, clean_loss, adv_loss


class TRADESLoss(nn.Module):
    """
    TRADES Loss: TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization

    Paper: "Theoretically Principled Trade-off between Robustness and Accuracy"
    https://arxiv.org/abs/1901.08573

    TRADES balances natural accuracy and robustness by:
    - Minimizing cross-entropy on clean examples
    - Minimizing KL divergence between clean and adversarial predictions

    Loss = CE(clean, labels) + beta * KL(adv || clean)

    Higher beta = more emphasis on robustness (may sacrifice clean accuracy)
    """

    def __init__(
        self,
        beta: float = 6.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.beta = beta

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        clean_logits: torch.Tensor,
        adv_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """
        Compute TRADES loss

        Args:
            clean_logits: Logits from clean examples
            adv_logits: Logits from adversarial examples
            targets: True labels

        Returns:
            total_loss, natural_loss, robust_loss
        """
        # Natural loss (cross-entropy on clean examples)
        natural_loss = self.ce_loss(clean_logits, targets)

        # Robust loss (KL divergence between clean and adversarial)
        # KL(adv || clean) = sum(adv * log(adv/clean))
        clean_probs = nn.functional.softmax(clean_logits, dim=1)
        adv_log_probs = nn.functional.log_softmax(adv_logits, dim=1)

        robust_loss = self.kl_loss(adv_log_probs, clean_probs)

        # Total TRADES loss
        total_loss = natural_loss + self.beta * robust_loss

        return total_loss, natural_loss, robust_loss


class MARTLoss(nn.Module):
    """
    MART Loss: Misclassification Aware adveRsarial Training

    Paper: "Improving Adversarial Robustness Requires Revisiting Misclassified Examples"
    https://openreview.net/forum?id=rklOg6EFwS

    MART focuses more on misclassified examples during adversarial training.
    """

    def __init__(
        self,
        beta: float = 6.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.beta = beta

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        self.kl_loss = nn.KLDivLoss(reduction='none')

    def forward(
        self,
        clean_logits: torch.Tensor,
        adv_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """
        Compute MART loss
        """
        batch_size = targets.size(0)

        # Cross-entropy on adversarial examples
        adv_ce = self.ce_loss(adv_logits, targets)

        # Get probabilities
        clean_probs = nn.functional.softmax(clean_logits, dim=1)
        adv_probs = nn.functional.softmax(adv_logits, dim=1)

        # Get the probability of true class for clean examples
        true_probs = clean_probs.gather(1, targets.unsqueeze(1)).squeeze()

        # KL divergence
        adv_log_probs = nn.functional.log_softmax(adv_logits, dim=1)
        kl = self.kl_loss(adv_log_probs, clean_probs).sum(dim=1)

        # MART loss: weight by (1 - p(y|x)) to focus on hard examples
        natural_loss = adv_ce.mean()
        robust_loss = (kl * (1 - true_probs)).mean()

        total_loss = natural_loss + self.beta * robust_loss

        return total_loss, natural_loss, robust_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Helps prevent overconfident predictions
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.class_weights = class_weights
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Create smooth labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Compute loss
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = (-smooth_labels * log_probs).sum(dim=-1) * weights
        else:
            loss = (-smooth_labels * log_probs).sum(dim=-1)

        return loss.mean()
