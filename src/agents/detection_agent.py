"""
Detection Agent
================
Role: Sentinel
Perception: Raw text inputs, model activation patterns, and feature distributions.
Action: Flag anomalous inputs and populate the adversarial candidate pool U.
Goal: Maximize adversarial recall while maintaining a low false-positive rate.

The Detection Agent serves as the perceptual gateway of the multi-agent system,
continuously monitoring incoming inputs to identify potential adversarial samples.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..models.classifier import CTIClassifier
from ..attacks.text_attacks import (
    TextAttacker, SynonymAttack, CharacterSwapAttack,
    HomoglyphAttack, KeyboardTypoAttack, CombinedAttack,
    RandomAttackSelector, BERTAttackSimulated
)
from ..attacks.fgsm import FGSMAttack


@dataclass
class AdversarialCandidate:
    """A candidate adversarial sample identified by the Detection Agent."""
    text: str
    original_text: str
    true_label: int
    label_name: str
    detection_method: str
    confidence_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


class DetectionAgent:
    """
    Detection Agent: Perceives incoming inputs and flags adversarial anomalies.

    Performs three key functions:
    1. Feature Consistency Analysis - checks for statistical deviations
    2. Prediction Confidence Analysis - flags low-confidence predictions
    3. Adversarial Pool Population - generates adversarial candidates using attacks
    """

    def __init__(
        self,
        classifier: CTIClassifier,
        id2label: Dict[int, str],
        attacks: Optional[Dict[str, TextAttacker]] = None,
        confidence_threshold: float = 0.7,
        seed: int = 42
    ):
        self.classifier = classifier
        self.id2label = id2label
        self.device = classifier.device
        self.confidence_threshold = confidence_threshold
        self.seed = seed

        # Initialize attack suite — diverse attacks for multi-attack pool
        if attacks is None:
            self.attacks = {
                "synonym": SynonymAttack(num_replacements=5, seed=seed),
                "char_swap": CharacterSwapAttack(num_swaps=3, seed=seed),
                "homoglyph": HomoglyphAttack(num_replacements=4, seed=seed),
                "keyboard_typo": KeyboardTypoAttack(num_typos=3, seed=seed),
                "bert_attack": BERTAttackSimulated(num_replacements=5, seed=seed),
                "combined": CombinedAttack(seed=seed),
            }
        else:
            self.attacks = attacks

        # Detection statistics
        self.stats = {
            "total_scanned": 0,
            "total_flagged": 0,
            "flagged_by_method": {},
        }

    def analyze_prediction_confidence(
        self,
        text: str,
        max_length: int = 512
    ) -> Tuple[int, float, np.ndarray]:
        """
        Analyze model confidence for a single input.

        Returns:
            (predicted_label, max_confidence, probability_distribution)
        """
        probs = self.classifier.get_probabilities(text, max_length=max_length)
        probs_np = probs.cpu().numpy()
        pred_label = int(np.argmax(probs_np))
        max_conf = float(np.max(probs_np))
        return pred_label, max_conf, probs_np

    def compute_prediction_entropy(self, probs: np.ndarray) -> float:
        """
        Compute prediction entropy H(x) = -sum(p * log(p)).

        High entropy = model is uncertain = potentially adversarial or informative.
        """
        # Clip to avoid log(0)
        probs_clipped = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))
        return float(entropy)

    def generate_adversarial_pool(
        self,
        df: pd.DataFrame,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
        pool_size: int = 500,
        random_state: int = 42
    ) -> List[AdversarialCandidate]:
        """
        Generate the adversarial candidate pool U by applying text-level attacks.

        For each sample, applies multiple attack strategies and checks if the
        model's prediction changes or confidence drops.

        Args:
            df: DataFrame with text and label columns
            text_column: Column name for text
            label_column: Column name for encoded labels
            pool_size: Maximum number of candidates to generate
            random_state: Random seed for sampling

        Returns:
            List of AdversarialCandidate objects forming pool U
        """
        # Sample if needed
        if pool_size < len(df):
            df_sample = df.sample(n=pool_size, random_state=random_state)
        else:
            df_sample = df

        candidate_pool: List[AdversarialCandidate] = []
        self.stats["total_scanned"] = 0

        for idx, row in df_sample.iterrows():
            original_text = row[text_column]
            true_label = int(row[label_column])
            label_name = self.id2label.get(true_label, str(true_label))
            self.stats["total_scanned"] += 1

            # Get original prediction and confidence
            orig_pred, orig_conf, orig_probs = self.analyze_prediction_confidence(original_text)

            # Apply each attack and check if it causes misclassification or drops confidence
            for attack_name, attacker in self.attacks.items():
                adv_text = attacker(original_text)
                adv_pred, adv_conf, adv_probs = self.analyze_prediction_confidence(adv_text)

                # Flag as candidate if:
                # 1. Originally correct prediction becomes wrong, OR
                # 2. Confidence drops significantly
                is_flagged = False

                if orig_pred == true_label and adv_pred != true_label:
                    # Successful attack - definitely add to pool
                    is_flagged = True
                    detection_method = f"{attack_name}_misclassification"
                elif adv_conf < self.confidence_threshold:
                    # Low confidence on adversarial input
                    is_flagged = True
                    detection_method = f"{attack_name}_low_confidence"

                if is_flagged:
                    entropy = self.compute_prediction_entropy(adv_probs)
                    candidate = AdversarialCandidate(
                        text=adv_text,
                        original_text=original_text,
                        true_label=true_label,
                        label_name=label_name,
                        detection_method=detection_method,
                        confidence_score=adv_conf,
                        metadata={
                            "original_pred": orig_pred,
                            "adversarial_pred": adv_pred,
                            "original_confidence": orig_conf,
                            "adversarial_confidence": adv_conf,
                            "entropy": entropy,
                            "attack_type": attack_name,
                        }
                    )
                    candidate_pool.append(candidate)
                    self.stats["total_flagged"] += 1
                    self.stats["flagged_by_method"][attack_name] = (
                        self.stats["flagged_by_method"].get(attack_name, 0) + 1
                    )

                    # Only keep one adversarial variant per sample per attack
                    break

        print(f"Detection Agent: Scanned {self.stats['total_scanned']} samples, "
              f"flagged {len(candidate_pool)} adversarial candidates")

        return candidate_pool

    def generate_fgsm_pool(
        self,
        df: pd.DataFrame,
        epsilon: float = 0.1,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
        pool_size: int = 500,
        random_state: int = 42
    ) -> List[AdversarialCandidate]:
        """
        Generate adversarial candidates via FGSM embedding attacks.
        These are the samples that actually shift the decision boundary.
        The text is preserved but marked with FGSM metadata for retraining
        with embedding perturbations.
        """
        if pool_size < len(df):
            df_sample = df.sample(n=pool_size, random_state=random_state)
        else:
            df_sample = df

        fgsm = FGSMAttack(
            model=self.classifier.get_model(),
            tokenizer=self.classifier.get_tokenizer(),
            epsilon=epsilon,
            device=self.device,
        )

        candidates = []
        for idx, row in df_sample.iterrows():
            text = row[text_column]
            true_label = int(row[label_column])
            label_name = self.id2label.get(true_label, str(true_label))

            try:
                orig_pred, adv_pred, attack_success = fgsm.attack_text(
                    text, true_label, max_length=256
                )
            except Exception:
                continue

            if attack_success:
                # Get confidence info
                probs = self.classifier.get_probabilities(text, max_length=256)
                probs_np = probs.cpu().numpy()
                conf = float(np.max(probs_np))
                entropy = self.compute_prediction_entropy(probs_np)

                candidates.append(AdversarialCandidate(
                    text=text,  # Original text — FGSM is applied at embedding level during retraining
                    original_text=text,
                    true_label=true_label,
                    label_name=label_name,
                    detection_method="fgsm_embedding_attack",
                    confidence_score=conf,
                    metadata={
                        "original_pred": orig_pred,
                        "adversarial_pred": adv_pred,
                        "confidence": conf,
                        "entropy": entropy,
                        "attack_type": "fgsm",
                        "epsilon": epsilon,
                    }
                ))

        print(f"Detection Agent (FGSM): {len(candidates)} embedding-attack candidates from {len(df_sample)} samples")
        return candidates

    def flag_low_confidence_inputs(
        self,
        df: pd.DataFrame,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
    ) -> List[AdversarialCandidate]:
        """
        Flag natural inputs where the model has low prediction confidence.
        These represent samples near decision boundaries.
        """
        candidates = []

        for idx, row in df.iterrows():
            text = row[text_column]
            true_label = int(row[label_column])
            label_name = self.id2label.get(true_label, str(true_label))

            pred, conf, probs = self.analyze_prediction_confidence(text)
            entropy = self.compute_prediction_entropy(probs)

            if conf < self.confidence_threshold:
                candidate = AdversarialCandidate(
                    text=text,
                    original_text=text,
                    true_label=true_label,
                    label_name=label_name,
                    detection_method="low_confidence_natural",
                    confidence_score=conf,
                    metadata={
                        "prediction": pred,
                        "confidence": conf,
                        "entropy": entropy,
                        "is_natural": True,
                    }
                )
                candidates.append(candidate)

        return candidates

    def get_statistics(self) -> Dict:
        """Return detection statistics."""
        return self.stats.copy()
