"""
Evaluator Module
Model evaluation and adversarial robustness testing
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from ..models.classifier import CTIClassifier
from ..attacks.text_attacks import TextAttacker, CombinedAttack
from ..attacks.fgsm import FGSMAttack
from .metrics import (
    compute_classification_metrics,
    compute_robustness_metrics,
    compute_attack_metrics
)


class ModelEvaluator:
    """Evaluator for CTI classification models"""

    def __init__(
        self,
        classifier: CTIClassifier,
        id2label: Optional[Dict[int, str]] = None
    ):
        self.classifier = classifier
        self.model = classifier.get_model()
        self.tokenizer = classifier.get_tokenizer()
        self.device = classifier.device
        self.id2label = id2label or {}

    def evaluate_loader(
        self,
        loader: DataLoader,
        desc: str = "Evaluating"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate model on a DataLoader

        Returns:
            Tuple of (true_labels, predictions)
        """
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
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

        return np.array(true_labels), np.array(predictions)

    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on a dataset

        Returns:
            Dictionary of metrics
        """
        y_true, y_pred = self.evaluate_loader(loader)

        label_names = None
        if self.id2label:
            label_names = [self.id2label[i] for i in range(len(self.id2label))]

        metrics = compute_classification_metrics(y_true, y_pred, label_names)

        return metrics

    def get_predictions_with_probs(
        self,
        loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions with probability scores

        Returns:
            Tuple of (true_labels, predictions, probabilities)
        """
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Getting predictions"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return (
            np.array(true_labels),
            np.array(predictions),
            np.array(probabilities)
        )


class AdversarialEvaluator:
    """Evaluator for adversarial robustness"""

    def __init__(
        self,
        classifier: CTIClassifier,
        attack: Optional[TextAttacker] = None,
        id2label: Optional[Dict[int, str]] = None
    ):
        self.classifier = classifier
        self.model = classifier.get_model()
        self.tokenizer = classifier.get_tokenizer()
        self.device = classifier.device
        self.id2label = id2label or {}

        # Default attack if none provided
        self.attack = attack or CombinedAttack()

    def evaluate_text_attacks(
        self,
        df: pd.DataFrame,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
        n_samples: Optional[int] = None,
        random_state: int = 42
    ) -> List[Dict]:
        """
        Evaluate model against text-level attacks

        Args:
            df: DataFrame with texts and labels
            text_column: Name of text column
            label_column: Name of label column
            n_samples: Number of samples to evaluate (None = all)
            random_state: Random seed for sampling

        Returns:
            List of attack results
        """
        if n_samples is not None and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=random_state)

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Attacking"):
            original_text = row[text_column]
            true_label = int(row[label_column])

            # Generate adversarial text
            adv_text = self.attack(original_text)

            # Get predictions
            orig_pred = self.classifier.predict(original_text)
            adv_pred = self.classifier.predict(adv_text)

            # Check attack success
            attack_success = (orig_pred == true_label) and (adv_pred != true_label)

            results.append({
                "original_text": original_text[:200] + "...",  # Truncate for storage
                "adversarial_text": adv_text[:200] + "...",
                "label": true_label,
                "label_name": self.id2label.get(true_label, str(true_label)),
                "original_pred": orig_pred,
                "adv_pred": adv_pred,
                "attack_success": attack_success,
                "originally_correct": orig_pred == true_label,
                "still_correct": adv_pred == true_label
            })

        return results

    def evaluate_fgsm_attacks(
        self,
        df: pd.DataFrame,
        epsilon: float = 0.1,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
        n_samples: Optional[int] = None,
        random_state: int = 42
    ) -> List[Dict]:
        """
        Evaluate model against FGSM attacks

        Args:
            df: DataFrame with texts and labels
            epsilon: FGSM perturbation strength
            text_column: Name of text column
            label_column: Name of label column
            n_samples: Number of samples to evaluate
            random_state: Random seed

        Returns:
            List of attack results
        """
        fgsm = FGSMAttack(
            model=self.model,
            tokenizer=self.tokenizer,
            epsilon=epsilon,
            device=self.device
        )

        if n_samples is not None and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=random_state)

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="FGSM Attack"):
            text = row[text_column]
            true_label = int(row[label_column])

            orig_pred, adv_pred, attack_success = fgsm.attack_text(
                text, true_label
            )

            results.append({
                "label": true_label,
                "label_name": self.id2label.get(true_label, str(true_label)),
                "original_pred": orig_pred,
                "adv_pred": adv_pred,
                "attack_success": attack_success,
                "originally_correct": orig_pred == true_label,
                "still_correct": adv_pred == true_label
            })

        return results

    def compare_attacks(
        self,
        df: pd.DataFrame,
        attacks: Dict[str, TextAttacker],
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
        n_samples: int = 100,
        random_state: int = 42
    ) -> Dict[str, Dict]:
        """
        Compare multiple attack methods

        Args:
            df: DataFrame with texts and labels
            attacks: Dictionary of attack name -> attacker
            text_column: Name of text column
            label_column: Name of label column
            n_samples: Number of samples
            random_state: Random seed

        Returns:
            Dictionary of attack results
        """
        # Sample once for fair comparison
        if n_samples < len(df):
            df = df.sample(n=n_samples, random_state=random_state)

        comparison = {}

        for attack_name, attacker in attacks.items():
            print(f"\nEvaluating {attack_name} attack...")
            self.attack = attacker

            results = self.evaluate_text_attacks(
                df,
                text_column=text_column,
                label_column=label_column,
                n_samples=None  # Already sampled
            )

            metrics = compute_attack_metrics(results)
            comparison[attack_name] = metrics

        return comparison

    def get_robustness_summary(
        self,
        attack_results: List[Dict]
    ) -> Dict:
        """
        Get summary of robustness evaluation

        Args:
            attack_results: List of attack result dictionaries

        Returns:
            Summary metrics
        """
        metrics = compute_attack_metrics(attack_results)

        # Add per-class breakdown
        class_results = {}
        for result in attack_results:
            label = result["label_name"]
            if label not in class_results:
                class_results[label] = {
                    "total": 0,
                    "originally_correct": 0,
                    "still_correct": 0,
                    "attacks_successful": 0
                }

            class_results[label]["total"] += 1
            if result["originally_correct"]:
                class_results[label]["originally_correct"] += 1
            if result["still_correct"]:
                class_results[label]["still_correct"] += 1
            if result["attack_success"]:
                class_results[label]["attacks_successful"] += 1

        # Compute per-class ASR
        for label in class_results:
            cr = class_results[label]
            if cr["originally_correct"] > 0:
                cr["asr"] = cr["attacks_successful"] / cr["originally_correct"]
            else:
                cr["asr"] = 0.0

        metrics["per_class"] = class_results

        return metrics
