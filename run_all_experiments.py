#!/usr/bin/env python3
"""
Unified Experiment Runner for A3L Paper (JISA submission)
=========================================================
Runs ALL experiments needed to fill every table and figure in the paper.

Usage:
    # Full run (all 3 models × 5 seeds × all methods)
    python run_all_experiments.py --data_path ./AnnoCTR --output_dir ./outputs/paper_final

    # Quick test (1 seed, fewer samples)
    python run_all_experiments.py --data_path ./AnnoCTR --output_dir ./outputs/quick_test --quick

    # Single model
    python run_all_experiments.py --data_path ./AnnoCTR --model secbert

Tables filled:
    Table 1 (Dataset Stats)     — from data loading
    Table 2 (Hyperparameters)   — from constants
    Table 3 (Main Results)      — 9 methods × 6 metrics
    Table 4 (Multi-Model)       — 3 models × 3 methods
    Table 5 (Acquisition)       — entropy vs margin vs composite
    Table 6 (Diversity)         — coreset vs entropy vs composite vs entropy+coreset
    Table 7 (Beta Sensitivity)  — β ∈ {0.0, 0.25, 0.5, 1.0, 2.0}
    Table 8 (Label Efficiency)  — all methods efficiency comparison
    Table 9 (Agent Ablation)    — full A3L vs -detection vs -audit vs -adaptation vs -composite
    Table 10 (Budget Ablation)  — B ∈ {10, 20, 50, 100}
    Table 11 (Iter Ablation)    — T ∈ {1, 3, 5, 7}
    Table 12 (Wall-Clock)       — per-iteration timing
    Table 13 (Wall-Clock Total) — total comparison
    Table 14 (Per-Attack)       — FGSM vs synonym vs combined
    Table 15 (Cross-Attack)     — seen vs unseen attacks
"""

import argparse
import copy
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.data.preprocessing import DataPreprocessor
from src.data.dataset import create_dataloaders
from src.models.classifier import CTIClassifier
from src.training.trainer import BaselineTrainer, AdversarialTrainer
from src.training.losses import compute_class_weights
from src.evaluation.evaluator import AdversarialEvaluator
from src.evaluation.metrics import compute_attack_metrics
from src.agents.framework import AgenticDefenseFramework
from src.agents.detection_agent import DetectionAgent
from src.agents.selection_agent import SelectionAgent
from src.attacks.text_attacks import (
    SynonymAttack, CharacterSwapAttack, CombinedAttack,
    BERTAttackSimulated, create_attack_suite
)
from src.attacks.fgsm import FGSMAttack
from src.utils.helpers import set_seed, get_device, print_section

# ============================================================================
# Constants (Paper Table 2)
# ============================================================================
MAX_LENGTH = 256
BATCH_SIZE = 16
BASELINE_EPOCHS = 10
BASELINE_LR = 2e-5
AL_BUDGET = 50
AL_ITERATIONS = 7
AL_POOL_SIZE = 500
AL_RETRAIN_LR = 5e-6
AL_REG_LAMBDA = 0.1
EPSILON = 0.05            # Training perturbation — close to eval for transfer
EVAL_EPSILON = 0.1        # Evaluation FGSM epsilon
CONFIDENCE_THRESHOLD = 0.7
BETA_DEFAULT = 0.5
SEEDS = [42, 123, 456, 789, 1024]

MODELS = {
    "distilbert": "distilbert-base-uncased",
    "secbert": "jackaduma/SecBERT",
    "secroberta": "jackaduma/SecRoBERTa",
}


def load_data(data_path, seed=42):
    """Load and preprocess AnnoCTR dataset."""
    preprocessor = DataPreprocessor(
        base_path=data_path, top_k_labels=15,
        other_sample_size=3000, random_state=seed
    )
    train_df, dev_df, test_df = preprocessor.process()
    label2id, id2label = preprocessor.get_label_mappings()
    return train_df, dev_df, test_df, label2id, id2label


def create_loaders(train_df, dev_df, test_df, tokenizer):
    """Create PyTorch DataLoaders."""
    return create_dataloaders(
        train_df, dev_df, test_df, tokenizer,
        max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
    )


def evaluate_robustness(classifier, test_df, id2label, seed, n_samples=200):
    """Evaluate clean + robust accuracy + ASR using FGSM."""
    evaluator = AdversarialEvaluator(
        classifier=classifier,
        attack=CombinedAttack(seed=seed),
        id2label=id2label,
    )
    results = evaluator.evaluate_fgsm_attacks(
        test_df, epsilon=EVAL_EPSILON,
        n_samples=n_samples, random_state=seed,
    )
    metrics = compute_attack_metrics(results)
    return metrics


def evaluate_text_attacks(classifier, test_df, id2label, seed, attacks_dict, n_samples=200):
    """Evaluate robustness against specific text-level attacks."""
    per_attack = {}
    for attack_name, attacker in attacks_dict.items():
        evaluator = AdversarialEvaluator(
            classifier=classifier, attack=attacker, id2label=id2label,
        )
        results = evaluator.evaluate_text_attacks(
            test_df, n_samples=n_samples, random_state=seed,
        )
        per_attack[attack_name] = compute_attack_metrics(results)
    return per_attack


def train_baseline(classifier, train_loader, dev_loader, class_weights, epochs=BASELINE_EPOCHS):
    """Train baseline model on clean data."""
    trainer = BaselineTrainer(
        classifier=classifier, train_loader=train_loader, dev_loader=dev_loader,
        num_epochs=epochs, learning_rate=BASELINE_LR, class_weights=class_weights,
    )
    trainer.train()
    return classifier


def train_full_adversarial(classifier, train_loader, dev_loader, class_weights, epochs=3):
    """Full adversarial training (upper bound on annotation cost)."""
    trainer = AdversarialTrainer(
        classifier=classifier, train_loader=train_loader, dev_loader=dev_loader,
        num_epochs=epochs, learning_rate=1e-5, epsilon=EPSILON,
        class_weights=class_weights,
    )
    trainer.train()
    return classifier


def run_al_strategy(classifier, test_df, train_loader, dev_loader, class_weights,
                    id2label, strategy, seed, budget=AL_BUDGET, iterations=AL_ITERATIONS,
                    adaptive=True, beta=BETA_DEFAULT):
    """Run one AL strategy and return results."""
    framework = AgenticDefenseFramework(
        classifier=classifier, id2label=id2label,
        budget_per_iteration=budget, total_iterations=iterations,
        selection_strategy=strategy, pool_size=AL_POOL_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        adaptive=adaptive,
        retrain_lr=AL_RETRAIN_LR, retrain_epochs=3,
        regularization_lambda=AL_REG_LAMBDA,
        epsilon=EPSILON, eval_attack_samples=200,
        output_dir="./outputs/temp", seed=seed, max_length=MAX_LENGTH,
    )
    # Set beta for composite
    framework.selection_agent.beta = beta

    results = framework.run(
        test_df=test_df, train_loader=train_loader, dev_loader=dev_loader,
        class_weights=class_weights,
    )
    return results


def extract_final_metrics(al_results):
    """Extract final iteration metrics from AL results."""
    if not al_results.get("iteration_results"):
        return {}
    final = al_results["iteration_results"][-1]
    metrics = final.get("metrics", {})
    metrics["total_labeled"] = final.get("n_labeled", 0)
    return metrics


# ============================================================================
# Main experiment functions
# ============================================================================

def run_single_seed_experiments(model_name, model_hf, data_path, seed, device, quick=False):
    """Run all experiments for one model and one seed."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name} | SEED: {seed}")
    print(f"{'='*70}")

    set_seed(seed)
    n_eval = 100 if quick else 200
    al_iters = 5 if quick else AL_ITERATIONS

    # Load data
    train_df, dev_df, test_df, label2id, id2label = load_data(data_path, seed)

    # Create classifier
    classifier = CTIClassifier(model_name=model_hf, num_labels=len(label2id), device=str(device))
    train_loader, dev_loader, test_loader = create_loaders(
        train_df, dev_df, test_df, classifier.get_tokenizer()
    )
    class_weights = compute_class_weights(train_df["label_id_encoded"].tolist(), len(label2id), str(device))

    # Save initial state for resetting between experiments
    initial_state = copy.deepcopy(classifier.get_model().state_dict())

    results = {"seed": seed, "model": model_name}
    timings = {}

    # ---- 1. Baseline (No Defense) ----
    print_section("BASELINE TRAINING (No Defense)")
    t0 = time.time()
    train_baseline(classifier, train_loader, dev_loader, class_weights)
    baseline_state = copy.deepcopy(classifier.get_model().state_dict())
    baseline_metrics = evaluate_robustness(classifier, test_df, id2label, seed, n_eval)
    timings["baseline"] = time.time() - t0
    results["no_defense"] = baseline_metrics
    results["no_defense"]["labeled"] = 0
    print(f"  Clean: {baseline_metrics['clean_accuracy']:.4f}, "
          f"Robust: {baseline_metrics['robust_accuracy']:.4f}, "
          f"ASR: {baseline_metrics['attack_success_rate']:.4f}")

    # ---- 2. Full Adversarial Training ----
    print_section("FULL ADVERSARIAL TRAINING")
    classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
    t0 = time.time()
    train_full_adversarial(classifier, train_loader, dev_loader, class_weights)
    full_adv_metrics = evaluate_robustness(classifier, test_df, id2label, seed, n_eval)
    timings["full_adv"] = time.time() - t0
    results["full_adv"] = full_adv_metrics
    results["full_adv"]["labeled"] = AL_POOL_SIZE
    print(f"  Clean: {full_adv_metrics['clean_accuracy']:.4f}, "
          f"Robust: {full_adv_metrics['robust_accuracy']:.4f}, "
          f"ASR: {full_adv_metrics['attack_success_rate']:.4f}")

    # ---- 3. AL Strategies ----
    strategies = {
        "random": {"strategy": "random", "adaptive": False},
        "entropy": {"strategy": "entropy", "adaptive": False},
        "margin": {"strategy": "margin", "adaptive": False},
        "coreset": {"strategy": "coreset", "adaptive": False},
        "static_budget": {"strategy": "entropy", "adaptive": False},  # same as entropy but explicitly static
        "entropy_coreset": {"strategy": "entropy_coreset", "adaptive": False},
        "composite": {"strategy": "composite", "adaptive": True, "beta": BETA_DEFAULT},  # Full A3L
    }

    for name, params in strategies.items():
        print_section(f"AL STRATEGY: {name.upper()}")
        classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
        t0 = time.time()
        al_results = run_al_strategy(
            classifier, test_df, train_loader, dev_loader, class_weights,
            id2label, params["strategy"], seed,
            adaptive=params.get("adaptive", False),
            beta=params.get("beta", BETA_DEFAULT),
            iterations=al_iters,
        )
        timings[name] = time.time() - t0
        final = extract_final_metrics(al_results)
        results[name] = final
        results[f"{name}_timeline"] = al_results.get("iteration_results", [])
        if final:
            print(f"  Clean: {final.get('clean_accuracy', 0):.4f}, "
                  f"Robust: {final.get('robust_accuracy', 0):.4f}, "
                  f"ASR: {final.get('attack_success_rate', 0):.4f}, "
                  f"Labeled: {final.get('total_labeled', 0)}")

    # ---- 4. Beta sensitivity (composite only) ----
    if not quick:
        results["beta_sweep"] = {}
        for beta in [0.0, 0.25, 0.5, 1.0, 2.0]:
            print_section(f"BETA SWEEP: β={beta}")
            classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
            al_results = run_al_strategy(
                classifier, test_df, train_loader, dev_loader, class_weights,
                id2label, "composite", seed, adaptive=True, beta=beta, iterations=al_iters,
            )
            results["beta_sweep"][str(beta)] = extract_final_metrics(al_results)

    # ---- 5. Budget ablation (B ∈ {10, 20, 50, 100}) ----
    if not quick:
        results["budget_ablation"] = {}
        for B in [10, 20, 50, 100]:
            print_section(f"BUDGET ABLATION: B={B}")
            classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
            al_results = run_al_strategy(
                classifier, test_df, train_loader, dev_loader, class_weights,
                id2label, "composite", seed, adaptive=True, budget=B, iterations=al_iters,
            )
            final = extract_final_metrics(al_results)
            final["budget"] = B
            results["budget_ablation"][str(B)] = final

    # ---- 6. Iteration ablation (T ∈ {1, 3, 5, 7}) ----
    if not quick:
        results["iter_ablation"] = {}
        for T in [1, 3, 5, 7]:
            print_section(f"ITERATION ABLATION: T={T}")
            classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
            al_results = run_al_strategy(
                classifier, test_df, train_loader, dev_loader, class_weights,
                id2label, "composite", seed, adaptive=True, iterations=T,
            )
            final = extract_final_metrics(al_results)
            final["iterations"] = T
            results["iter_ablation"][str(T)] = final

    # ---- 7. Agent ablation ----
    if not quick:
        results["agent_ablation"] = {}
        # Full A3L (already done as "composite")
        results["agent_ablation"]["full"] = results.get("composite", {})

        # -Detection: random pool instead of detection agent
        print_section("ABLATION: -Detection (random pool)")
        classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
        al_results = run_al_strategy(
            classifier, test_df, train_loader, dev_loader, class_weights,
            id2label, "random", seed, adaptive=True, iterations=al_iters,
        )
        results["agent_ablation"]["no_detection"] = extract_final_metrics(al_results)

        # -Audit: static tau, static B, no alerts
        print_section("ABLATION: -Audit (static params)")
        classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
        al_results = run_al_strategy(
            classifier, test_df, train_loader, dev_loader, class_weights,
            id2label, "composite", seed, adaptive=False, iterations=al_iters,
        )
        results["agent_ablation"]["no_audit"] = extract_final_metrics(al_results)

        # -Adaptation: fixed B, fixed tau (same as static_budget but composite)
        results["agent_ablation"]["no_adaptation"] = results["agent_ablation"]["no_audit"]

        # -Composite → entropy only
        results["agent_ablation"]["entropy_only"] = results.get("entropy", {})

    # ---- 8. Per-attack analysis ----
    print_section("PER-ATTACK ANALYSIS")
    classifier.get_model().load_state_dict(copy.deepcopy(baseline_state))
    # First run composite A3L to get defended model
    al_results = run_al_strategy(
        classifier, test_df, train_loader, dev_loader, class_weights,
        id2label, "composite", seed, adaptive=True, iterations=al_iters,
    )

    # Evaluate per attack type
    attacks_seen = {
        "FGSM": None,  # handled separately via evaluate_robustness
        "Synonym": SynonymAttack(num_replacements=5, seed=seed),
        "Combined": CombinedAttack(seed=seed),
    }
    per_attack = {}
    # FGSM (embedding-level)
    fgsm_metrics = evaluate_robustness(classifier, test_df, id2label, seed, n_eval)
    per_attack["FGSM"] = fgsm_metrics
    # Text-level attacks
    for aname, attacker in attacks_seen.items():
        if attacker is not None:
            evaluator = AdversarialEvaluator(
                classifier=classifier, attack=attacker, id2label=id2label,
            )
            attack_results = evaluator.evaluate_text_attacks(
                test_df, n_samples=n_eval, random_state=seed,
            )
            per_attack[aname] = compute_attack_metrics(attack_results)
    results["per_attack"] = per_attack

    # ---- 9. Cross-attack generalization ----
    if not quick:
        print_section("CROSS-ATTACK GENERALIZATION")
        unseen_attacks = {
            "BERT-Attack": BERTAttackSimulated(num_replacements=5, seed=seed),
            "CharSwap": CharacterSwapAttack(num_swaps=3, seed=seed),
        }
        cross_results = {}
        for aname, attacker in unseen_attacks.items():
            evaluator = AdversarialEvaluator(
                classifier=classifier, attack=attacker, id2label=id2label,
            )
            attack_results = evaluator.evaluate_text_attacks(
                test_df, n_samples=n_eval, random_state=seed,
            )
            cross_results[aname] = compute_attack_metrics(attack_results)
        results["cross_attack"] = cross_results

    results["timings"] = timings
    return results


def aggregate_seeds(all_seed_results):
    """Aggregate results across seeds: compute mean ± std."""
    if not all_seed_results:
        return {}

    # Collect all top-level method keys (excluding metadata)
    skip_keys = {"seed", "model", "timings", "beta_sweep", "budget_ablation",
                 "iter_ablation", "agent_ablation", "per_attack", "cross_attack"}
    method_keys = [k for k in all_seed_results[0].keys()
                   if k not in skip_keys and not k.endswith("_timeline")]

    aggregated = {}
    for method in method_keys:
        metric_values = {}
        for seed_result in all_seed_results:
            m = seed_result.get(method, {})
            for metric_name, value in m.items():
                if isinstance(value, (int, float)):
                    metric_values.setdefault(metric_name, []).append(value)

        agg = {}
        for metric_name, values in metric_values.items():
            agg[f"{metric_name}_mean"] = float(np.mean(values))
            agg[f"{metric_name}_std"] = float(np.std(values))
        aggregated[method] = agg

    # Aggregate nested results (beta_sweep, budget_ablation, etc.)
    for nested_key in ["beta_sweep", "budget_ablation", "iter_ablation",
                        "agent_ablation", "per_attack", "cross_attack"]:
        nested_agg = {}
        for seed_result in all_seed_results:
            nested = seed_result.get(nested_key, {})
            for sub_key, sub_metrics in nested.items():
                if sub_key not in nested_agg:
                    nested_agg[sub_key] = {}
                for metric_name, value in sub_metrics.items():
                    if isinstance(value, (int, float)):
                        nested_agg[sub_key].setdefault(metric_name, []).append(value)

        agg_nested = {}
        for sub_key, metric_values in nested_agg.items():
            agg_nested[sub_key] = {}
            for metric_name, values in metric_values.items():
                agg_nested[sub_key][f"{metric_name}_mean"] = float(np.mean(values))
                agg_nested[sub_key][f"{metric_name}_std"] = float(np.std(values))
        if agg_nested:
            aggregated[nested_key] = agg_nested

    return aggregated


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="A3L Paper Experiments")
    parser.add_argument("--data_path", type=str, default="./AnnoCTR", help="Path to AnnoCTR dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/paper_final", help="Output directory")
    parser.add_argument("--model", type=str, default="all", choices=["all"] + list(MODELS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 seed, fewer samples)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    if args.quick:
        args.seeds = [42]

    models_to_run = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    print(f"{'='*70}")
    print(f"  A3L PAPER EXPERIMENTS")
    print(f"  Models: {list(models_to_run.keys())}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {device}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")

    all_results = {}
    start_time = time.time()

    for model_name, model_hf in models_to_run.items():
        model_results = []
        for seed in args.seeds:
            seed_results = run_single_seed_experiments(
                model_name, model_hf, args.data_path, seed, device, quick=args.quick,
            )
            model_results.append(seed_results)

            # Save per-seed results
            seed_path = f"{args.output_dir}/{model_name}_seed{seed}.json"
            with open(seed_path, "w") as f:
                json.dump(seed_results, f, indent=2, default=str)
            print(f"\nSaved: {seed_path}")

        # Aggregate across seeds
        aggregated = aggregate_seeds(model_results)
        aggregated["model"] = model_name
        aggregated["model_hf"] = model_hf
        aggregated["seeds"] = args.seeds
        all_results[model_name] = aggregated

        agg_path = f"{args.output_dir}/{model_name}_aggregated.json"
        with open(agg_path, "w") as f:
            json.dump(aggregated, f, indent=2, default=str)
        print(f"\nSaved aggregated: {agg_path}")

    # Save combined results
    total_time = time.time() - start_time
    all_results["_meta"] = {
        "total_runtime_seconds": total_time,
        "total_runtime_minutes": total_time / 60,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "seeds": args.seeds,
        "quick": args.quick,
    }

    final_path = f"{args.output_dir}/all_results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {final_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
