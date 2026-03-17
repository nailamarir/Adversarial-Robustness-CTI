"""
Metrics Module
Evaluation metrics for classification and robustness
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names for reporting

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    if labels is not None:
        for i, label in enumerate(labels):
            if i < len(f1_per_class):
                metrics[f"f1_{label}"] = f1_per_class[i]

    return metrics


def compute_robustness_metrics(
    original_preds: np.ndarray,
    adversarial_preds: np.ndarray,
    true_labels: np.ndarray
) -> Dict:
    """
    Compute adversarial robustness metrics

    Args:
        original_preds: Predictions on clean data
        adversarial_preds: Predictions on adversarial data
        true_labels: True labels

    Returns:
        Dictionary of robustness metrics
    """
    n_samples = len(true_labels)

    # Clean accuracy
    clean_correct = (original_preds == true_labels)
    clean_accuracy = np.mean(clean_correct)

    # Robust accuracy
    robust_correct = (adversarial_preds == true_labels)
    robust_accuracy = np.mean(robust_correct)

    # Attack success rate (flip from correct to incorrect)
    originally_correct = clean_correct
    flipped_to_wrong = (adversarial_preds != true_labels) & originally_correct

    n_originally_correct = np.sum(originally_correct)
    n_successful_attacks = np.sum(flipped_to_wrong)

    if n_originally_correct > 0:
        attack_success_rate = n_successful_attacks / n_originally_correct
    else:
        attack_success_rate = 0.0

    # Prediction stability (how often predictions change)
    prediction_changed = (original_preds != adversarial_preds)
    stability = 1 - np.mean(prediction_changed)

    # Accuracy drop
    accuracy_drop = clean_accuracy - robust_accuracy

    # Relative accuracy drop
    if clean_accuracy > 0:
        relative_accuracy_drop = accuracy_drop / clean_accuracy
    else:
        relative_accuracy_drop = 0.0

    return {
        "clean_accuracy": clean_accuracy,
        "robust_accuracy": robust_accuracy,
        "attack_success_rate": attack_success_rate,
        "prediction_stability": stability,
        "accuracy_drop": accuracy_drop,
        "relative_accuracy_drop": relative_accuracy_drop,
        "n_samples": n_samples,
        "n_originally_correct": int(n_originally_correct),
        "n_successful_attacks": int(n_successful_attacks)
    }


def compute_attack_metrics(
    attack_results: List[Dict]
) -> Dict:
    """
    Compute metrics from attack results

    Args:
        attack_results: List of attack result dictionaries

    Returns:
        Aggregated metrics
    """
    n_total = len(attack_results)

    if n_total == 0:
        return {
            "n_samples": 0,
            "clean_accuracy": 0.0,
            "robust_accuracy": 0.0,
            "attack_success_rate": 0.0
        }

    # Extract values
    original_correct = sum(
        1 for r in attack_results if r["original_pred"] == r["label"]
    )
    robust_correct = sum(
        1 for r in attack_results if r["adv_pred"] == r["label"]
    )
    successful_attacks = sum(
        1 for r in attack_results if r["attack_success"]
    )

    clean_accuracy = original_correct / n_total
    robust_accuracy = robust_correct / n_total

    if original_correct > 0:
        asr = successful_attacks / original_correct
    else:
        asr = 0.0

    return {
        "n_samples": n_total,
        "n_originally_correct": original_correct,
        "n_robust_correct": robust_correct,
        "n_successful_attacks": successful_attacks,
        "clean_accuracy": clean_accuracy,
        "robust_accuracy": robust_accuracy,
        "attack_success_rate": asr,
        "robustness_gap": clean_accuracy - robust_accuracy
    }


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    output_dict: bool = True
) -> Dict:
    """
    Generate detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names for each class
        output_dict: Return as dictionary

    Returns:
        Classification report
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=output_dict,
        zero_division=0
    )
    return report


def compute_confusion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Compute confusion matrix based metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix and derived metrics
    """
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics from confusion matrix
    n_classes = cm.shape[0]

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0)
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), 0)

    return {
        "confusion_matrix": cm,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_specificity": specificity
    }


def compute_label_efficiency(
    robustness_timeline: List[Dict]
) -> Dict:
    """
    Compute label efficiency metrics from the active learning robustness timeline.

    Label Efficiency (Eq. 11 in paper):
        Label Efficiency = Delta Robust Accuracy / Number of Labeled Samples

    This measures how effectively the labeling budget translates to robustness gains.

    Args:
        robustness_timeline: List of dicts with iteration, robust_accuracy, n_labeled_samples

    Returns:
        Dictionary of label efficiency metrics
    """
    if len(robustness_timeline) < 2:
        return {"label_efficiency": 0.0}

    baseline_robust = robustness_timeline[0].get("robust_accuracy", 0)
    efficiencies = []
    per_iteration = []

    for i, entry in enumerate(robustness_timeline[1:], 1):
        robust_acc = entry.get("robust_accuracy", 0)
        n_labeled = entry.get("n_labeled_samples", 0)
        delta_robust = robust_acc - baseline_robust

        eff = delta_robust / max(n_labeled, 1)
        efficiencies.append(eff)

        # Per-iteration marginal efficiency
        if i > 1:
            prev_robust = robustness_timeline[i - 1].get("robust_accuracy", 0)
            prev_labeled = robustness_timeline[i - 1].get("n_labeled_samples", 0)
            marginal_delta = robust_acc - prev_robust
            marginal_samples = n_labeled - prev_labeled
            marginal_eff = marginal_delta / max(marginal_samples, 1)
        else:
            marginal_eff = eff

        per_iteration.append({
            "iteration": i,
            "robust_accuracy": robust_acc,
            "n_labeled": n_labeled,
            "delta_robust": delta_robust,
            "cumulative_efficiency": eff,
            "marginal_efficiency": marginal_eff,
        })

    return {
        "label_efficiency": float(np.mean(efficiencies)) if efficiencies else 0.0,
        "final_efficiency": efficiencies[-1] if efficiencies else 0.0,
        "max_efficiency": float(np.max(efficiencies)) if efficiencies else 0.0,
        "per_iteration": per_iteration,
        "total_delta_robust": float(
            robustness_timeline[-1].get("robust_accuracy", 0) - baseline_robust
        ),
        "total_labeled": robustness_timeline[-1].get("n_labeled_samples", 0),
    }


def compare_al_strategies(
    strategy_results: Dict[str, List[Dict]]
) -> Dict:
    """
    Compare multiple active learning strategies.

    Args:
        strategy_results: Dict mapping strategy name -> robustness timeline

    Returns:
        Comparison dictionary
    """
    comparison = {}
    for strategy_name, timeline in strategy_results.items():
        if len(timeline) < 2:
            continue
        eff = compute_label_efficiency(timeline)
        comparison[strategy_name] = {
            "final_robust_accuracy": timeline[-1].get("robust_accuracy", 0),
            "final_clean_accuracy": timeline[-1].get("clean_accuracy", 0),
            "final_asr": timeline[-1].get("attack_success_rate", 0),
            "total_labeled": timeline[-1].get("n_labeled_samples", 0),
            "label_efficiency": eff["label_efficiency"],
            "total_delta_robust": eff["total_delta_robust"],
        }
    return comparison


def compare_model_architectures(
    model_results: Dict[str, Dict]
) -> Dict:
    """
    Compare multiple model architectures (e.g., DistilBERT vs SecBERT).

    Args:
        model_results: Dict mapping model name -> metrics dict with keys like
                       clean_accuracy, robust_accuracy, attack_success_rate, f1_macro, etc.

    Returns:
        Comparison dictionary with per-model metrics and a 'best' summary.
    """
    comparison = {}
    for model_name, metrics in model_results.items():
        comparison[model_name] = {
            "clean_accuracy": metrics.get("clean_accuracy", 0),
            "robust_accuracy": metrics.get("robust_accuracy", 0),
            "attack_success_rate": metrics.get("attack_success_rate", 0),
            "f1_macro": metrics.get("f1_macro", 0),
            "f1_weighted": metrics.get("f1_weighted", 0),
            "total_params": metrics.get("total_params", "N/A"),
        }

    # Determine best model per metric
    if len(comparison) >= 2:
        best = {}
        for metric in ["clean_accuracy", "robust_accuracy", "f1_macro", "f1_weighted"]:
            best[metric] = max(comparison, key=lambda m: comparison[m].get(metric, 0))
        # Lower is better for ASR
        best["attack_success_rate"] = min(
            comparison, key=lambda m: comparison[m].get("attack_success_rate", 1)
        )
        comparison["best_per_metric"] = best

    return comparison


def compare_models(
    baseline_metrics: Dict,
    adversarial_metrics: Dict
) -> Dict:
    """
    Compare baseline and adversarially trained model metrics

    Args:
        baseline_metrics: Metrics from baseline model
        adversarial_metrics: Metrics from adversarial model

    Returns:
        Comparison dictionary with improvements
    """
    comparison = {}

    # Compare common metrics
    for key in baseline_metrics:
        if key in adversarial_metrics:
            baseline_val = baseline_metrics[key]
            adv_val = adversarial_metrics[key]

            if isinstance(baseline_val, (int, float)):
                comparison[key] = {
                    "baseline": baseline_val,
                    "adversarial": adv_val,
                    "difference": adv_val - baseline_val,
                }

                # Relative improvement
                if baseline_val != 0:
                    comparison[key]["relative_change"] = (
                        (adv_val - baseline_val) / abs(baseline_val)
                    )

    # Key improvement metrics
    if "attack_success_rate" in comparison:
        baseline_asr = comparison["attack_success_rate"]["baseline"]
        adv_asr = comparison["attack_success_rate"]["adversarial"]

        if baseline_asr > 0:
            asr_reduction = (baseline_asr - adv_asr) / baseline_asr * 100
        else:
            asr_reduction = 0.0

        comparison["asr_reduction_percent"] = asr_reduction

    return comparison
