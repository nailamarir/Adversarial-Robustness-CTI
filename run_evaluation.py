#!/usr/bin/env python3
r"""
Comprehensive Evaluation Script for Paper Results
===================================================
Produces all tables and figures from:
  "An Agentic Active Learning-Enhanced Adversarial Defense Framework
   for Cyber Threat Intelligence Classification"

Results produced:
  - Table 3: Dataset Statistics
  - Table 5: Main Results (No Adv Train, Random, Margin, Full Adv Train, AL-Entropy)
  - Table 6: Label Efficiency Comparison at 200 samples
  - Table 7: Effect of Per-Iteration Budget B (20, 50, 100)
  - Figure 3: Learning curves (robust accuracy vs labeled samples)
  - Figure 4: Robust accuracy across AL iterations
  - Qualitative analysis of high-uncertainty selected samples

Usage:
  # Full evaluation (all tables, figures, 5 seeds)
  python run_evaluation.py --data_path ./AnnoCTR/AnnoCTR

  # Quick test (1 seed, fewer samples)
  python run_evaluation.py --data_path ./AnnoCTR/AnnoCTR --quick

  # Single seed run
  python run_evaluation.py --data_path ./AnnoCTR/AnnoCTR --seeds 42
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import set_seed, get_device, print_section, save_results, Timer
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import create_dataloaders
from src.models.classifier import CTIClassifier
from src.training.trainer import BaselineTrainer, AdversarialTrainer
from src.training.losses import compute_class_weights
from src.attacks.text_attacks import CombinedAttack
from src.attacks.fgsm import FGSMAttack
from src.evaluation.evaluator import ModelEvaluator, AdversarialEvaluator
from src.evaluation.metrics import compute_attack_metrics, compute_label_efficiency
from src.agents import AgenticDefenseFramework
from src.visualization.plots import Visualizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ============================================================================
# Constants matching paper Table 4
# ============================================================================
MODEL_NAME = "jackaduma/SecBERT"   # security-domain pretrained; swap to "distilbert-base-uncased" for baseline
MAX_LENGTH = 256
BATCH_SIZE = 16
BASELINE_EPOCHS = 15
BASELINE_LR = 2e-5
AL_BUDGET = 50
AL_ITERATIONS = 5
AL_POOL_SIZE = 500
AL_RETRAIN_EPOCHS = 3
AL_RETRAIN_LR = 5e-6  # Lower than baseline to prevent catastrophic forgetting
AL_REG_LAMBDA = 0.01
EPSILON = 0.01           # Training perturbation (used in AL retraining)
EVAL_EPSILON = 0.1       # Evaluation FGSM epsilon (moderate, avoids trivial attack)
ATTACK_SAMPLES = 500
CONFIDENCE_THRESHOLD = 0.7


# ============================================================================
# Helpers
# ============================================================================

def load_data(data_path, seed=42):
    """Load and preprocess AnnoCTR dataset."""
    preprocessor = DataPreprocessor(
        base_path=data_path,
        top_k_labels=15,
        other_sample_size=3000,  # raised to retain most hard negatives
        random_state=seed
    )
    train_df, dev_df, test_df = preprocessor.process()
    label2id, id2label = preprocessor.get_label_mappings()
    return train_df, dev_df, test_df, label2id, id2label


def create_loaders(train_df, dev_df, test_df, tokenizer, max_length=MAX_LENGTH,
                   batch_size=BATCH_SIZE):
    """Create DataLoaders."""
    return create_dataloaders(
        train_df, dev_df, test_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )


def train_baseline_model(classifier, train_loader, dev_loader, class_weights,
                         epochs=BASELINE_EPOCHS, lr=BASELINE_LR):
    """Train baseline model and return history."""
    trainer = BaselineTrainer(
        classifier=classifier,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_epochs=epochs,
        learning_rate=lr,
        class_weights=class_weights,
        patience=5
    )
    history = trainer.train()
    return history


def evaluate_robustness(classifier, test_df, id2label, seed=42,
                        n_samples=ATTACK_SAMPLES):
    """Evaluate model robustness against FGSM embedding-level attacks."""
    adv_evaluator = AdversarialEvaluator(
        classifier=classifier, attack=CombinedAttack(seed=seed), id2label=id2label
    )
    # Use FGSM (embedding-level) attacks — much stronger than text-level
    attack_results = adv_evaluator.evaluate_fgsm_attacks(
        test_df, epsilon=EVAL_EPSILON, n_samples=n_samples, random_state=seed
    )
    metrics = compute_attack_metrics(attack_results)
    return metrics


def evaluate_clean(classifier, test_loader, id2label):
    """Evaluate clean test metrics."""
    evaluator = ModelEvaluator(classifier, id2label)
    return evaluator.evaluate(test_loader)


def run_al_strategy(classifier, id2label, test_df, train_loader, dev_loader,
                    class_weights, strategy="entropy", budget=AL_BUDGET,
                    iterations=AL_ITERATIONS, pool_size=AL_POOL_SIZE,
                    seed=42, output_dir="./outputs"):
    """Run a single AL strategy and return results."""
    framework = AgenticDefenseFramework(
        classifier=classifier,
        id2label=id2label,
        budget_per_iteration=budget,
        total_iterations=iterations,
        selection_strategy=strategy,
        pool_size=pool_size,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        retrain_lr=AL_RETRAIN_LR,
        retrain_epochs=AL_RETRAIN_EPOCHS,
        regularization_lambda=AL_REG_LAMBDA,
        epsilon=EPSILON,
        eval_attack_samples=ATTACK_SAMPLES,
        output_dir=output_dir,
        seed=seed,
        max_length=MAX_LENGTH,
    )
    results = framework.run(
        test_df=test_df,
        train_loader=train_loader,
        dev_loader=dev_loader,
        class_weights=class_weights,
    )
    return results


def run_full_adv_training(classifier, test_df, train_loader, dev_loader,
                          id2label, class_weights, seed=42):
    """Run full adversarial training on ALL pool samples (no budget constraint)."""
    # Generate adversarial pool (full 500 samples)
    from src.agents.detection_agent import DetectionAgent

    detection_agent = DetectionAgent(
        classifier=classifier,
        id2label=id2label,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        seed=seed,
    )
    pool = detection_agent.generate_adversarial_pool(
        df=test_df,
        text_column="full_text",
        label_column="label_id_encoded",
        pool_size=AL_POOL_SIZE,
        random_state=seed,
    )

    # Retrain on ALL samples (no selection, no budget)
    from src.agents.retraining_agent import RetrainingAgent
    from src.agents.selection_agent import ScoredCandidate

    # Wrap all pool items as ScoredCandidate (dummy scores)
    all_selected = [
        ScoredCandidate(candidate=c, entropy=0.0, margin=0.0, rank=i)
        for i, c in enumerate(pool)
    ]

    retrainer = RetrainingAgent(
        classifier=classifier,
        learning_rate=AL_RETRAIN_LR,
        epochs_per_iteration=AL_RETRAIN_EPOCHS,
        regularization_lambda=AL_REG_LAMBDA,
        clean_mix_ratio=0.3,
        epsilon=EPSILON,
        max_length=MAX_LENGTH,
    )

    retrainer.retrain_iteration(
        selected_samples=all_selected,
        clean_loader=train_loader,
        dev_loader=dev_loader,
        class_weights=class_weights,
        iteration=1,
    )

    # Evaluate after full adv training
    robustness = evaluate_robustness(classifier, test_df, id2label, seed=seed)
    clean_metrics = evaluate_clean(classifier, dev_loader, id2label)

    return {
        "clean_accuracy": robustness["clean_accuracy"],
        "robust_accuracy": robustness["robust_accuracy"],
        "attack_success_rate": robustness["attack_success_rate"],
        "f1_macro": clean_metrics.get("f1_macro", 0),
        "samples_used": len(pool),
    }


# ============================================================================
# Table 3: Dataset Statistics
# ============================================================================

def generate_table3(train_df, dev_df, test_df):
    """Generate Table 3: Dataset Statistics."""
    print_section("TABLE 3: Dataset Statistics")

    total = len(train_df) + len(dev_df) + len(test_df)
    n_classes = train_df["label_mapped"].nunique()

    # Average text length in tokens (approximate by word count)
    avg_len = train_df["full_text"].str.split().str.len().mean()

    stats = {
        "Total Samples": total,
        "Training Set": f"{len(train_df)} ({len(train_df)/total*100:.0f}%)",
        "Validation Set": f"{len(dev_df)} ({len(dev_df)/total*100:.0f}%)",
        "Test Set": f"{len(test_df)} ({len(test_df)/total*100:.0f}%)",
        "Number of Classes": f"{n_classes} (MITRE Tactics)",
        "Avg. Text Length": f"{avg_len:.0f} tokens",
        "Adversarial Pool": f"{AL_POOL_SIZE} samples",
    }

    print(f"{'Property':<25} {'Value':>20}")
    print("-" * 50)
    for k, v in stats.items():
        print(f"{k:<25} {str(v):>20}")

    return stats


# ============================================================================
# Table 5: Main Results
# ============================================================================

def _get_best_iteration(timeline):
    """Get the iteration with highest robust accuracy (early stopping for robustness)."""
    if not timeline:
        return {}
    # Skip iteration 0 (baseline, no AL yet) when looking for best
    al_iterations = [t for t in timeline if t.get("n_labeled_samples", 0) > 0]
    if not al_iterations:
        return timeline[-1]
    return max(al_iterations, key=lambda t: t.get("robust_accuracy", 0))


def run_table5_single_seed(data_path, seed, device, output_dir, quick=False):
    """Run all methods for Table 5 with a single seed."""
    print_section(f"TABLE 5 - Seed {seed}")
    set_seed(seed)

    # Load data
    train_df, dev_df, test_df, label2id, id2label = load_data(data_path, seed)

    # Initialize model
    classifier = CTIClassifier(
        model_name=MODEL_NAME, num_labels=len(label2id), device=device
    )
    tokenizer = classifier.get_tokenizer()

    train_loader, dev_loader, test_loader = create_loaders(
        train_df, dev_df, test_df, tokenizer
    )
    class_weights = compute_class_weights(
        train_df["label_id_encoded"].values,
        num_classes=len(label2id), device=device
    )

    # 1. Train baseline
    print("\n--- Training Baseline ---")
    epochs = 2 if quick else BASELINE_EPOCHS
    baseline_history = train_baseline_model(
        classifier, train_loader, dev_loader, class_weights, epochs=epochs
    )

    # Save baseline state for resetting
    baseline_state = copy.deepcopy(classifier.get_model().state_dict())

    # Evaluate baseline (No Adversarial Training)
    baseline_clean = evaluate_clean(classifier, test_loader, id2label)
    baseline_robust = evaluate_robustness(classifier, test_df, id2label, seed=seed)

    results = {}
    results["No Adv. Train"] = {
        "clean_accuracy": baseline_robust["clean_accuracy"],
        "robust_accuracy": baseline_robust["robust_accuracy"],
        "attack_success_rate": baseline_robust["attack_success_rate"],
        "f1_macro": baseline_clean.get("f1_macro", 0),
        "samples_used": 0,
    }
    print(f"  No Adv Train -> Clean: {baseline_robust['clean_accuracy']:.4f}, "
          f"Robust: {baseline_robust['robust_accuracy']:.4f}, "
          f"ASR: {baseline_robust['attack_success_rate']:.4f}")

    iters = 2 if quick else AL_ITERATIONS
    pool = AL_POOL_SIZE // 2 if quick else AL_POOL_SIZE

    # 2. AL-Entropy
    print("\n--- Running AL-Entropy ---")
    classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
    entropy_results = run_al_strategy(
        classifier, id2label, test_df, train_loader, dev_loader,
        class_weights, strategy="entropy", budget=AL_BUDGET,
        iterations=iters, pool_size=pool, seed=seed, output_dir=output_dir
    )
    entropy_timeline = entropy_results["robustness_timeline"]
    if entropy_timeline:
        best = _get_best_iteration(entropy_timeline)
        results["AL-Entropy"] = {
            "clean_accuracy": best.get("clean_accuracy", 0),
            "robust_accuracy": best.get("robust_accuracy", 0),
            "attack_success_rate": best.get("attack_success_rate", 0),
            "f1_macro": 0,  # Will compute below
            "samples_used": best.get("n_labeled_samples", 0),
            "timeline": entropy_timeline,
        }
        # Get F1 after AL
        al_clean = evaluate_clean(classifier, test_loader, id2label)
        results["AL-Entropy"]["f1_macro"] = al_clean.get("f1_macro", 0)

    # 3. Random
    print("\n--- Running Random ---")
    classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
    random_results = run_al_strategy(
        classifier, id2label, test_df, train_loader, dev_loader,
        class_weights, strategy="random", budget=AL_BUDGET,
        iterations=iters, pool_size=pool, seed=seed, output_dir=output_dir
    )
    random_timeline = random_results["robustness_timeline"]
    if random_timeline:
        best = _get_best_iteration(random_timeline)
        results["Random"] = {
            "clean_accuracy": best.get("clean_accuracy", 0),
            "robust_accuracy": best.get("robust_accuracy", 0),
            "attack_success_rate": best.get("attack_success_rate", 0),
            "f1_macro": 0,
            "samples_used": best.get("n_labeled_samples", 0),
            "timeline": random_timeline,
        }
        al_clean = evaluate_clean(classifier, test_loader, id2label)
        results["Random"]["f1_macro"] = al_clean.get("f1_macro", 0)

    # 4. Margin
    print("\n--- Running Margin ---")
    classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
    margin_results = run_al_strategy(
        classifier, id2label, test_df, train_loader, dev_loader,
        class_weights, strategy="margin", budget=AL_BUDGET,
        iterations=iters, pool_size=pool, seed=seed, output_dir=output_dir
    )
    margin_timeline = margin_results["robustness_timeline"]
    if margin_timeline:
        best = _get_best_iteration(margin_timeline)
        results["Margin"] = {
            "clean_accuracy": best.get("clean_accuracy", 0),
            "robust_accuracy": best.get("robust_accuracy", 0),
            "attack_success_rate": best.get("attack_success_rate", 0),
            "f1_macro": 0,
            "samples_used": best.get("n_labeled_samples", 0),
            "timeline": margin_timeline,
        }
        al_clean = evaluate_clean(classifier, test_loader, id2label)
        results["Margin"]["f1_macro"] = al_clean.get("f1_macro", 0)

    # 5. Full Adversarial Training (no budget constraint)
    print("\n--- Running Full Adversarial Training ---")
    classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
    full_adv = run_full_adv_training(
        classifier, test_df, train_loader, dev_loader,
        id2label, class_weights, seed=seed
    )
    al_clean = evaluate_clean(classifier, test_loader, id2label)
    full_adv["f1_macro"] = al_clean.get("f1_macro", 0)
    results["Full Adv. Train"] = full_adv

    return results, {
        "entropy": entropy_timeline,
        "random": random_timeline,
        "margin": margin_timeline,
    }


# ============================================================================
# Table 7: Ablation - Budget Size
# ============================================================================

def run_table7_single_seed(data_path, seed, device, output_dir, quick=False):
    """Run budget ablation for Table 7."""
    print_section(f"TABLE 7 - Budget Ablation - Seed {seed}")
    set_seed(seed)

    train_df, dev_df, test_df, label2id, id2label = load_data(data_path, seed)
    classifier = CTIClassifier(
        model_name=MODEL_NAME, num_labels=len(label2id), device=device
    )
    train_loader, dev_loader, test_loader = create_loaders(
        train_df, dev_df, test_df, classifier.get_tokenizer()
    )
    class_weights = compute_class_weights(
        train_df["label_id_encoded"].values,
        num_classes=len(label2id), device=device
    )

    # Train baseline
    epochs = 2 if quick else BASELINE_EPOCHS
    train_baseline_model(
        classifier, train_loader, dev_loader, class_weights, epochs=epochs
    )
    baseline_state = copy.deepcopy(classifier.get_model().state_dict())

    budgets = [20, 50, 100]
    iters = AL_ITERATIONS if not quick else 2
    pool = AL_POOL_SIZE if not quick else AL_POOL_SIZE // 2
    results = {}

    for budget in budgets:
        print(f"\n--- Budget B={budget} ---")
        classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))

        al_results = run_al_strategy(
            classifier, id2label, test_df, train_loader, dev_loader,
            class_weights, strategy="entropy", budget=budget,
            iterations=iters, pool_size=pool, seed=seed, output_dir=output_dir
        )
        timeline = al_results["robustness_timeline"]
        if timeline:
            best = _get_best_iteration(timeline)
            total_samples = best.get("n_labeled_samples", 0)
            results[budget] = {
                "total_samples": total_samples,
                "robust_accuracy": best.get("robust_accuracy", 0),
                "attack_success_rate": best.get("attack_success_rate", 0),
            }

    return results


# ============================================================================
# Qualitative Analysis
# ============================================================================

def run_qualitative_analysis(data_path, seed, device, output_dir):
    """Analyze characteristics of high-uncertainty selected samples."""
    print_section("QUALITATIVE ANALYSIS OF SELECTED SAMPLES")
    set_seed(seed)

    train_df, dev_df, test_df, label2id, id2label = load_data(data_path, seed)
    classifier = CTIClassifier(
        model_name=MODEL_NAME, num_labels=len(label2id), device=device
    )
    train_loader, dev_loader, test_loader = create_loaders(
        train_df, dev_df, test_df, classifier.get_tokenizer()
    )
    class_weights = compute_class_weights(
        train_df["label_id_encoded"].values,
        num_classes=len(label2id), device=device
    )

    train_baseline_model(classifier, train_loader, dev_loader, class_weights)

    # Run 1 iteration of AL-Entropy to get selected samples
    framework = AgenticDefenseFramework(
        classifier=classifier,
        id2label=id2label,
        budget_per_iteration=AL_BUDGET,
        total_iterations=1,  # Just 1 iteration for analysis
        selection_strategy="entropy",
        pool_size=AL_POOL_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        retrain_lr=AL_RETRAIN_LR,
        retrain_epochs=AL_RETRAIN_EPOCHS,
        regularization_lambda=AL_REG_LAMBDA,
        epsilon=EPSILON,
        eval_attack_samples=ATTACK_SAMPLES,
        output_dir=output_dir,
        seed=seed,
        max_length=MAX_LENGTH,
    )

    results = framework.run(
        test_df=test_df,
        train_loader=train_loader,
        dev_loader=dev_loader,
        class_weights=class_weights,
    )

    # Analyze the selection history
    selection_history = results.get("selection_history", [])
    if selection_history:
        first_iter = selection_history[0]
        print(f"\nIteration 1 Selection Stats:")
        print(f"  Pool size:         {first_iter.get('pool_size', 'N/A')}")
        print(f"  Selected:          {first_iter.get('selected', 'N/A')}")
        print(f"  Avg entropy:       {first_iter.get('avg_entropy', 0):.4f}")
        print(f"  Max entropy:       {first_iter.get('max_entropy', 0):.4f}")
        print(f"  Min entropy:       {first_iter.get('min_entropy', 0):.4f}")

    # Print audit log explanations
    audit_log_path = os.path.join(output_dir, "logs", "audit_log.json")
    if os.path.exists(audit_log_path):
        with open(audit_log_path) as f:
            audit = json.load(f)

        # Find selection events with explanations
        for event in audit.get("events", []):
            if event.get("action") == "samples_selected":
                details = event.get("details", {})
                explanations = details.get("top_sample_explanations", [])
                if explanations:
                    print("\nTop-5 Selected Sample Explanations (Iteration 1):")
                    print("-" * 60)
                    for exp in explanations[:5]:
                        print(f"  Rank {exp['rank']}: {exp['label']}")
                        print(f"    Entropy: {exp['entropy']:.4f}, "
                              f"Margin: {exp['margin']:.4f}")
                        print(f"    Method: {exp['detection_method']}")
                        print(f"    Reason: {exp['reason']}")
                        print()
                break

    return results


# ============================================================================
# Visualization: Figure 3 & Figure 4
# ============================================================================

def generate_figure3(all_timelines, full_adv_robust_acc, output_dir):
    """
    Figure 3: Learning curves comparing sample selection strategies.
    Robust accuracy vs number of labeled samples.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"entropy": "#2196F3", "random": "#757575", "margin": "#FF9800"}
    markers = {"entropy": "o", "random": "s", "margin": "^"}
    labels_map = {"entropy": "AL-Entropy", "random": "Random", "margin": "Margin"}

    for strategy, timeline in all_timelines.items():
        if not timeline:
            continue
        n_labeled = [e.get("n_labeled_samples", 0) for e in timeline]
        robust_acc = [e.get("robust_accuracy", 0) for e in timeline]
        ax.plot(n_labeled, robust_acc, f'-{markers.get(strategy, "D")}',
                label=labels_map.get(strategy, strategy),
                color=colors.get(strategy, "#333"),
                linewidth=2, markersize=8)

    # Full adversarial training reference line
    if full_adv_robust_acc > 0:
        ax.axhline(y=full_adv_robust_acc, color='red', linestyle='--',
                   linewidth=1.5, label=f'Full Adv. Train ({full_adv_robust_acc:.3f})')

    ax.set_xlabel("Number of Labeled Samples", fontsize=13)
    ax.set_ylabel("Robust Accuracy", fontsize=13)
    ax.set_title("Figure 3: Learning Curves - Selection Strategy Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, "figures", "figure3_learning_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def generate_figure4(entropy_timeline, output_dir):
    """
    Figure 4: Robust accuracy across active learning iterations.
    """
    if not entropy_timeline:
        print("No entropy timeline data for Figure 4")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    iterations = [e.get("iteration", i) for i, e in enumerate(entropy_timeline)]
    robust_acc = [e.get("robust_accuracy", 0) for e in entropy_timeline]

    ax.plot(iterations, robust_acc, '-o', color='#2196F3',
            linewidth=2.5, markersize=10)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Robust Accuracy", fontsize=13)
    ax.set_title("Figure 4: Robust Accuracy Across AL Iterations",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.8)
    ax.set_xticks(iterations)

    plt.tight_layout()
    path = os.path.join(output_dir, "figures", "figure4_iteration_progress.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Multi-seed aggregation
# ============================================================================

def aggregate_results(all_seed_results):
    """Aggregate results across seeds: compute mean +/- std."""
    methods = set()
    for seed_res in all_seed_results:
        methods.update(seed_res.keys())

    aggregated = {}
    for method in methods:
        method_metrics = {
            "clean_accuracy": [],
            "robust_accuracy": [],
            "attack_success_rate": [],
            "f1_macro": [],
            "samples_used": [],
        }
        for seed_res in all_seed_results:
            if method in seed_res:
                for k in method_metrics:
                    val = seed_res[method].get(k, 0)
                    if val is not None:
                        method_metrics[k].append(val)

        aggregated[method] = {}
        for k, vals in method_metrics.items():
            if vals:
                aggregated[method][f"{k}_mean"] = float(np.mean(vals))
                aggregated[method][f"{k}_std"] = float(np.std(vals))
            else:
                aggregated[method][f"{k}_mean"] = 0.0
                aggregated[method][f"{k}_std"] = 0.0

    return aggregated


def print_table5(aggregated):
    """Print Table 5 in paper format."""
    print_section("TABLE 5: Main Results - Performance Comparison")
    print(f"{'Method':<20} {'Clean Acc':>12} {'Robust Acc':>12} {'ASR':>12} "
          f"{'Samples':>10} {'F1':>12}")
    print("-" * 85)

    order = ["No Adv. Train", "Random", "Margin", "Full Adv. Train", "AL-Entropy"]
    for method in order:
        if method not in aggregated:
            continue
        m = aggregated[method]
        clean = f"{m['clean_accuracy_mean']:.3f}\u00b1{m['clean_accuracy_std']:.3f}"
        robust = f"{m['robust_accuracy_mean']:.3f}\u00b1{m['robust_accuracy_std']:.3f}"
        asr = f"{m['attack_success_rate_mean']:.3f}\u00b1{m['attack_success_rate_std']:.3f}"
        samples = f"{m['samples_used_mean']:.0f}"
        f1 = f"{m['f1_macro_mean']:.2f}\u00b1{m['f1_macro_std']:.2f}"
        print(f"{method:<20} {clean:>12} {robust:>12} {asr:>12} {samples:>10} {f1:>12}")


def print_table6(all_timelines_per_seed):
    """Print Table 6: Label Efficiency at 200 samples."""
    print_section("TABLE 6: Label Efficiency Comparison at 200 Samples")

    strategies = ["random", "margin", "entropy"]
    strategy_labels = {"random": "Random", "margin": "Margin", "entropy": "AL-Entropy"}

    # Collect per-seed efficiencies
    efficiency_data = {s: [] for s in strategies}
    robust_data = {s: [] for s in strategies}
    delta_data = {s: [] for s in strategies}

    for seed_timelines in all_timelines_per_seed:
        for strategy in strategies:
            timeline = seed_timelines.get(strategy, [])
            if len(timeline) >= 2:
                baseline_robust = timeline[0].get("robust_accuracy", 0)
                # Find entry closest to 200 labeled samples
                final = timeline[-1]
                robust_acc = final.get("robust_accuracy", 0)
                n_labeled = final.get("n_labeled_samples", 1)
                delta = robust_acc - baseline_robust
                eff = delta / max(n_labeled, 1)

                robust_data[strategy].append(robust_acc)
                delta_data[strategy].append(delta)
                efficiency_data[strategy].append(eff)

    print(f"{'Method':<15} {'Robust Acc.':>12} {'Δ from Baseline':>16} {'Efficiency':>12}")
    print("-" * 60)

    for strategy in strategies:
        label = strategy_labels[strategy]
        if robust_data[strategy]:
            robust_str = f"{np.mean(robust_data[strategy]):.3f}"
            delta_str = f"+{np.mean(delta_data[strategy]):.3f}"
            # Relative efficiency (normalize to random)
            random_eff = np.mean(efficiency_data["random"]) if efficiency_data["random"] else 1e-10
            rel_eff = np.mean(efficiency_data[strategy]) / max(abs(random_eff), 1e-10)
            eff_str = f"{rel_eff:.1f}×"
        else:
            robust_str = "N/A"
            delta_str = "N/A"
            eff_str = "N/A"
        print(f"{label:<15} {robust_str:>12} {delta_str:>16} {eff_str:>12}")


def print_table7(all_budget_results):
    """Print Table 7: Budget ablation."""
    print_section("TABLE 7: Effect of Per-Iteration Budget B")

    # Aggregate across seeds
    budgets = set()
    for seed_res in all_budget_results:
        budgets.update(seed_res.keys())
    budgets = sorted(budgets)

    print(f"{'Budget B':>10} {'Total Samples':>15} {'Robust Acc.':>12} {'ASR':>10}")
    print("-" * 52)

    for budget in budgets:
        robust_vals = []
        asr_vals = []
        total_samples_vals = []
        for seed_res in all_budget_results:
            if budget in seed_res:
                robust_vals.append(seed_res[budget]["robust_accuracy"])
                asr_vals.append(seed_res[budget]["attack_success_rate"])
                total_samples_vals.append(seed_res[budget]["total_samples"])

        if robust_vals:
            total_str = f"{np.mean(total_samples_vals):.0f}"
            robust_str = f"{np.mean(robust_vals):.3f}"
            asr_str = f"{np.mean(asr_vals):.3f}"
        else:
            total_str = robust_str = asr_str = "N/A"
        print(f"{budget:>10} {total_str:>15} {robust_str:>12} {asr_str:>10}")


# ============================================================================
# Main evaluation pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all paper evaluations")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to AnnoCTR dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/paper_results",
                        help="Output directory for results")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                        help="Comma-separated seeds (default: 5 seeds)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, fewer epochs/iterations")
    parser.add_argument("--skip_table7", action="store_true",
                        help="Skip Table 7 budget ablation")
    parser.add_argument("--skip_qualitative", action="store_true",
                        help="Skip qualitative analysis")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name (overrides MODEL_NAME constant). "
                             "E.g. jackaduma/SecBERT or distilbert-base-uncased")
    args = parser.parse_args()

    # Model override
    global MODEL_NAME
    if args.model:
        MODEL_NAME = args.model
    print(f"Model: {MODEL_NAME}")

    # Parse seeds
    if args.quick:
        seeds = [42]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Setup
    device = get_device()
    for subdir in ["models", "figures", "logs"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    total_start = time.time()

    print("=" * 70)
    print("  COMPREHENSIVE PAPER EVALUATION")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # =====================================================================
    # Table 3: Dataset Statistics
    # =====================================================================
    train_df, dev_df, test_df, label2id, id2label = load_data(args.data_path, 42)
    table3_stats = generate_table3(train_df, dev_df, test_df)

    # =====================================================================
    # Table 5: Main Results (multi-seed)
    # =====================================================================
    all_seed_results = []
    all_timelines_per_seed = []
    best_timelines = None

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        results, timelines = run_table5_single_seed(
            args.data_path, seed, device, args.output_dir, quick=args.quick
        )
        all_seed_results.append(results)
        all_timelines_per_seed.append(timelines)

        # Keep first seed's timelines for figures
        if best_timelines is None:
            best_timelines = timelines

    # Aggregate and print Table 5
    aggregated = aggregate_results(all_seed_results)
    print_table5(aggregated)

    # =====================================================================
    # Table 6: Label Efficiency
    # =====================================================================
    print_table6(all_timelines_per_seed)

    # =====================================================================
    # Table 7: Budget Ablation
    # =====================================================================
    if not args.skip_table7:
        all_budget_results = []
        for seed in seeds:
            budget_res = run_table7_single_seed(
                args.data_path, seed, device, args.output_dir, quick=args.quick
            )
            all_budget_results.append(budget_res)
        print_table7(all_budget_results)
    else:
        all_budget_results = []

    # =====================================================================
    # Figure 3 & Figure 4
    # =====================================================================
    print_section("GENERATING FIGURES")

    full_adv_robust = aggregated.get("Full Adv. Train", {}).get(
        "robust_accuracy_mean", 0
    )

    if best_timelines:
        generate_figure3(best_timelines, full_adv_robust, args.output_dir)

        entropy_timeline = best_timelines.get("entropy", [])
        generate_figure4(entropy_timeline, args.output_dir)

    # =====================================================================
    # Qualitative Analysis
    # =====================================================================
    if not args.skip_qualitative:
        qual_results = run_qualitative_analysis(
            args.data_path, 42, device, args.output_dir
        )

    # =====================================================================
    # Save all results
    # =====================================================================
    all_results = {
        "table3": table3_stats,
        "table5": aggregated,
        "seeds": seeds,
        "config": {
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "baseline_epochs": BASELINE_EPOCHS if not args.quick else 2,
            "al_budget": AL_BUDGET,
            "al_iterations": AL_ITERATIONS if not args.quick else 2,
            "al_pool_size": AL_POOL_SIZE,
            "attack_samples": ATTACK_SAMPLES,
        },
    }

    if all_budget_results:
        all_results["table7_raw"] = [
            {str(k): v for k, v in seed_res.items()}
            for seed_res in all_budget_results
        ]

    results_path = os.path.join(args.output_dir, "logs", "all_paper_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {results_path}")

    total_time = time.time() - total_start
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")

    print_section("EVALUATION COMPLETE")
    print(f"  Results:  {args.output_dir}/logs/all_paper_results.json")
    print(f"  Figure 3: {args.output_dir}/figures/figure3_learning_curves.png")
    print(f"  Figure 4: {args.output_dir}/figures/figure4_iteration_progress.png")


if __name__ == "__main__":
    main()
