#!/usr/bin/env python3
"""
Agentic Active Learning-Enhanced Adversarial Defense Framework
for Cyber Threat Intelligence Classification
================================================================

Multi-agent system with four specialized agents:
  - Detection Agent: Flags adversarial anomalies and populates candidate pool
  - Selection Agent: Scores by uncertainty (entropy) and selects top-B samples
  - Retraining Agent: Incremental adversarial fine-tuning with regularization
  - Audit Agent: Monitors decisions, tracks robustness, generates explanations

Also supports standard adversarial training methods (FGSM, PGD, TRADES).

Usage:
    # Run the Agentic Active Learning framework (primary mode)
    python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic

    # Compare AL strategies (entropy vs margin vs random)
    python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic --al_compare

    # Compare DistilBERT vs SecBERT (baseline + adversarial robustness)
    python main.py --data_path ./AnnoCTR/AnnoCTR --model_compare

    # Compare models with agentic AL
    python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic --model_compare

    # Standard adversarial training (legacy mode)
    python main.py --data_path ./AnnoCTR/AnnoCTR --mode standard --method fgsm
    python main.py --data_path ./AnnoCTR/AnnoCTR --mode standard --method all
"""

import argparse
import os
import sys
import copy

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd

from src.utils.helpers import set_seed, get_device, print_section, save_results, Timer
from src.data.preprocessing import DataPreprocessor
from src.data.preprocessing_enhanced import EnhancedDataPreprocessor
from src.data.dataset import create_dataloaders
from src.models.classifier import CTIClassifier
from src.training.trainer import BaselineTrainer, AdversarialTrainer, PGDTrainer, TRADESTrainer
from src.training.losses import compute_class_weights
from src.attacks.text_attacks import (
    CombinedAttack, SynonymAttack, CharacterSwapAttack,
    HomoglyphAttack, KeyboardTypoAttack, create_attack_suite
)
from src.evaluation.evaluator import ModelEvaluator, AdversarialEvaluator
from src.evaluation.metrics import (
    compute_attack_metrics, compare_models, compare_model_architectures,
    compute_label_efficiency, compare_al_strategies
)
from src.visualization.plots import Visualizer, create_full_report
from src.agents import AgenticDefenseFramework
from configs.config import MODELS_TO_COMPARE


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Agentic Active Learning-Enhanced Adversarial Defense for CTI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Agentic Active Learning framework (paper's primary contribution)
  python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic

  # Compare AL strategies (entropy vs margin vs random)
  python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic --al_compare

  # Compare DistilBERT vs SecBERT
  python main.py --data_path ./AnnoCTR/AnnoCTR --model_compare
  python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic --model_compare

  # Agentic with custom budget and iterations
  python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic --al_budget 50 --al_iterations 5

  # Standard adversarial training (legacy)
  python main.py --data_path ./AnnoCTR/AnnoCTR --mode standard --method fgsm
  python main.py --data_path ./AnnoCTR/AnnoCTR --mode standard --method all

  # Quick test run
  python main.py --data_path ./AnnoCTR/AnnoCTR --mode agentic --epochs 2 --al_iterations 2 --al_budget 20
        """
    )

    # === Mode selection ===
    parser.add_argument(
        "--mode", type=str, default="agentic",
        choices=["agentic", "standard"],
        help="Framework mode: 'agentic' for AL defense (default), 'standard' for legacy adversarial training"
    )

    # === Data arguments ===
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to AnnoCTR dataset directory"
    )

    # === Active Learning arguments (agentic mode) ===
    parser.add_argument(
        "--al_budget", type=int, default=50,
        help="AL budget B: samples selected per iteration (default: 50)"
    )
    parser.add_argument(
        "--al_iterations", type=int, default=5,
        help="AL iterations T: number of active learning cycles (default: 5)"
    )
    parser.add_argument(
        "--al_strategy", type=str, default="entropy",
        choices=["entropy", "margin", "random"],
        help="AL selection strategy (default: entropy)"
    )
    parser.add_argument(
        "--al_pool_size", type=int, default=500,
        help="Size of adversarial candidate pool U (default: 500)"
    )
    parser.add_argument(
        "--al_compare", action="store_true",
        help="Compare all AL strategies (entropy, margin, random)"
    )
    parser.add_argument(
        "--model_compare", action="store_true",
        help="Compare DistilBERT vs SecBERT model architectures"
    )
    parser.add_argument(
        "--al_retrain_lr", type=float, default=2e-5,
        help="Learning rate for AL retraining (default: 2e-5)"
    )
    parser.add_argument(
        "--al_retrain_epochs", type=int, default=2,
        help="Epochs per AL retraining iteration (default: 2)"
    )
    parser.add_argument(
        "--al_reg_lambda", type=float, default=0.01,
        help="Regularization lambda for retraining (default: 0.01)"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.7,
        help="Detection agent confidence threshold (default: 0.7)"
    )

    # === Standard adversarial training arguments (standard mode) ===
    parser.add_argument(
        "--method", type=str, default="fgsm",
        choices=["fgsm", "pgd", "trades", "all"],
        help="Adversarial training method for standard mode (default: fgsm)"
    )
    parser.add_argument(
        "--adv_epochs", type=int, default=3,
        help="Number of adversarial training epochs (default: 3)"
    )
    parser.add_argument(
        "--adv_lr", type=float, default=1e-5,
        help="Learning rate for adversarial training (default: 1e-5)"
    )
    parser.add_argument(
        "--clean_weight", type=float, default=0.7,
        help="Weight for clean loss in FGSM (default: 0.7)"
    )
    parser.add_argument(
        "--pgd_steps", type=int, default=7,
        help="Number of PGD steps (default: 7)"
    )
    parser.add_argument(
        "--pgd_alpha", type=float, default=0.003,
        help="PGD step size (default: 0.003)"
    )
    parser.add_argument(
        "--trades_beta", type=float, default=6.0,
        help="TRADES beta (default: 6.0)"
    )
    parser.add_argument(
        "--trades_steps", type=int, default=7,
        help="Number of TRADES PGD steps (default: 7)"
    )

    # === Common arguments ===
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of baseline training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate for baseline training (default: 2e-5)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Perturbation epsilon (default: 0.01)"
    )
    parser.add_argument(
        "--model_name", type=str, default="distilbert-base-uncased",
        help="Pretrained model name (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Maximum sequence length (default: 256)"
    )
    parser.add_argument(
        "--top_k_labels", type=int, default=15,
        help="Keep top K most frequent labels (default: 15)"
    )
    parser.add_argument(
        "--other_sample_size", type=int, default=600,
        help="Number of OTHER samples to keep (default: 600)"
    )
    parser.add_argument(
        "--enhanced_preprocessing", action="store_true",
        help="Use enhanced preprocessing"
    )
    parser.add_argument(
        "--truncation_strategy", type=str, default="head_tail",
        choices=["head", "tail", "head_tail", "important"],
        help="Text truncation strategy (default: head_tail)"
    )
    parser.add_argument(
        "--no_augmentation", action="store_true",
        help="Disable data augmentation in enhanced preprocessing"
    )
    parser.add_argument(
        "--min_samples", type=int, default=50,
        help="Minimum samples per class"
    )
    parser.add_argument(
        "--max_samples", type=int, default=800,
        help="Maximum samples per class"
    )
    parser.add_argument(
        "--attack_samples", type=int, default=200,
        help="Number of samples for adversarial evaluation (default: 200)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip baseline training (load from checkpoint)"
    )
    parser.add_argument(
        "--baseline_checkpoint", type=str, default=None,
        help="Path to baseline checkpoint"
    )

    return parser.parse_args()


# ===========================================================================
# Standard adversarial training (legacy mode)
# ===========================================================================

def train_and_evaluate_method(
    method_name, classifier, train_loader, dev_loader, test_loader,
    test_df, args, class_weights, id2label, viz
):
    """Train with a specific adversarial method and evaluate."""
    print_section(f"Training with {method_name.upper()}")

    if method_name == "fgsm":
        trainer = AdversarialTrainer(
            classifier=classifier, train_loader=train_loader,
            dev_loader=dev_loader, num_epochs=args.adv_epochs,
            learning_rate=args.adv_lr, epsilon=args.epsilon,
            clean_weight=args.clean_weight, adv_weight=1 - args.clean_weight,
            class_weights=class_weights
        )
    elif method_name == "pgd":
        trainer = PGDTrainer(
            classifier=classifier, train_loader=train_loader,
            dev_loader=dev_loader, num_epochs=args.adv_epochs,
            learning_rate=args.adv_lr, epsilon=args.epsilon,
            alpha=args.pgd_alpha, num_steps=args.pgd_steps,
            class_weights=class_weights
        )
    elif method_name == "trades":
        trainer = TRADESTrainer(
            classifier=classifier, train_loader=train_loader,
            dev_loader=dev_loader, num_epochs=args.adv_epochs,
            learning_rate=args.adv_lr, epsilon=args.epsilon,
            step_size=args.pgd_alpha, num_steps=args.trades_steps,
            beta=args.trades_beta, class_weights=class_weights
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    with Timer(f"{method_name.upper()} training"):
        history = trainer.train()

    model_path = f"{args.output_dir}/models/{method_name}_model.pt"
    classifier.save(model_path)

    evaluator = ModelEvaluator(classifier, id2label)
    test_metrics = evaluator.evaluate(test_loader)
    print(f"\n{method_name.upper()} Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"{method_name.upper()} Test F1 (macro): {test_metrics['f1_macro']:.4f}")

    adv_evaluator = AdversarialEvaluator(
        classifier=classifier, attack=CombinedAttack(seed=args.seed),
        id2label=id2label
    )
    with Timer(f"{method_name.upper()} adversarial evaluation"):
        attack_results = adv_evaluator.evaluate_text_attacks(
            test_df, n_samples=args.attack_samples, random_state=args.seed
        )

    robustness = compute_attack_metrics(attack_results)
    print(f"\n{method_name.upper()} Robustness:")
    print(f"  Clean Accuracy:      {robustness['clean_accuracy']:.4f}")
    print(f"  Robust Accuracy:     {robustness['robust_accuracy']:.4f}")
    print(f"  Attack Success Rate: {robustness['attack_success_rate']:.4f}")

    y_true, y_pred = evaluator.evaluate_loader(test_loader, f"{method_name} predictions")

    return {
        "method": method_name, "history": history,
        "test_metrics": test_metrics, "robustness": robustness, "y_pred": y_pred
    }


# ===========================================================================
# Data preprocessing (shared)
# ===========================================================================

def load_and_preprocess(args):
    """Load and preprocess data. Returns (train_df, dev_df, test_df, label2id, id2label)."""
    with Timer("Data preprocessing"):
        if args.enhanced_preprocessing:
            print("Using ENHANCED preprocessing pipeline...")
            preprocessor = EnhancedDataPreprocessor(
                base_path=args.data_path,
                top_k_labels=args.top_k_labels,
                min_samples_per_class=args.min_samples,
                max_samples_per_class=args.max_samples,
                random_state=args.seed,
                clean_text=True,
                augment_data=not args.no_augmentation
            )
            train_df, dev_df, test_df = preprocessor.process(
                truncation_strategy=args.truncation_strategy
            )
        else:
            print("Using ORIGINAL preprocessing pipeline...")
            preprocessor = DataPreprocessor(
                base_path=args.data_path,
                top_k_labels=args.top_k_labels,
                other_sample_size=args.other_sample_size,
                random_state=args.seed
            )
            train_df, dev_df, test_df = preprocessor.process()

        label2id, id2label = preprocessor.get_label_mappings()

    print(f"\nDataset sizes: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}")
    print(f"Classes: {len(label2id)}")
    return train_df, dev_df, test_df, label2id, id2label


def train_baseline(classifier, train_loader, dev_loader, args, class_weights):
    """Train baseline model."""
    if args.skip_baseline and args.baseline_checkpoint:
        print(f"Loading baseline from: {args.baseline_checkpoint}")
        classifier.load(args.baseline_checkpoint)
        return {"train_loss": [], "dev_acc": [], "dev_f1": []}

    with Timer("Baseline training"):
        trainer = BaselineTrainer(
            classifier=classifier,
            train_loader=train_loader,
            dev_loader=dev_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            class_weights=class_weights,
            patience=5
        )
        history = trainer.train()

    classifier.save(f"{args.output_dir}/models/baseline_model.pt")
    return history


# ===========================================================================
# Agentic Active Learning Pipeline
# ===========================================================================

def run_agentic_mode(args, classifier, train_df, dev_df, test_df,
                     train_loader, dev_loader, test_loader,
                     label2id, id2label, class_weights, baseline_robustness,
                     baseline_test_metrics, baseline_history, y_true, y_pred_baseline):
    """Run the Agentic Active Learning-Enhanced Adaptive Defense Framework."""

    print_section("AGENTIC ACTIVE LEARNING DEFENSE FRAMEWORK")

    viz = Visualizer(save_dir=f"{args.output_dir}/figures")

    if args.al_compare:
        # =====================================================================
        # Compare multiple AL strategies (Section 5.5 of the paper)
        # =====================================================================
        print_section("COMPARING AL STRATEGIES: Entropy vs Margin vs Random")

        strategies = ["entropy", "margin", "random"]
        initial_state = copy.deepcopy(classifier.get_model().state_dict())
        all_strategy_results = {}
        all_timelines = {}

        for strategy in strategies:
            print(f"\n{'#'*60}")
            print(f"  STRATEGY: {strategy.upper()}")
            print(f"{'#'*60}")

            # Reset model
            classifier.get_model().load_state_dict(copy.deepcopy(initial_state))

            framework = AgenticDefenseFramework(
                classifier=classifier,
                id2label=id2label,
                budget_per_iteration=args.al_budget,
                total_iterations=args.al_iterations,
                selection_strategy=strategy,
                pool_size=args.al_pool_size,
                confidence_threshold=args.confidence_threshold,
                retrain_lr=args.al_retrain_lr,
                retrain_epochs=args.al_retrain_epochs,
                regularization_lambda=args.al_reg_lambda,
                epsilon=args.epsilon,
                eval_attack_samples=args.attack_samples,
                output_dir=args.output_dir,
                seed=args.seed,
                max_length=args.max_length,
            )

            results = framework.run(
                test_df=test_df,
                train_loader=train_loader,
                dev_loader=dev_loader,
                class_weights=class_weights,
            )

            all_strategy_results[strategy] = results
            all_timelines[strategy] = results["robustness_timeline"]

        # Restore initial state
        classifier.get_model().load_state_dict(initial_state)

        # Compare strategies
        strategy_comparison = compare_al_strategies(all_timelines)

        # Generate comparison visualizations
        print_section("GENERATING AL COMPARISON VISUALIZATIONS")

        viz.plot_al_learning_curves(
            all_timelines,
            title="AL Learning Curves: Robust Accuracy vs Labeled Samples",
            save_name="al_learning_curves"
        )

        viz.plot_al_strategy_comparison(
            strategy_comparison,
            title="AL Strategy Comparison (Table 5/6)",
            save_name="al_strategy_comparison"
        )

        # Plot iteration progress for best strategy (entropy)
        if "entropy" in all_timelines:
            viz.plot_al_iteration_progress(
                all_timelines["entropy"],
                title="AL-Entropy: Accuracy Across Iterations",
                save_name="al_entropy_iterations"
            )
            viz.plot_asr_over_iterations(
                all_timelines["entropy"],
                title="AL-Entropy: ASR Reduction Over Iterations",
                save_name="al_entropy_asr"
            )

        # Save comparison results
        save_results(
            {
                "mode": "agentic_comparison",
                "strategies": list(all_strategy_results.keys()),
                "comparison": strategy_comparison,
                "timelines": all_timelines,
                "baseline": {
                    "test_metrics": baseline_test_metrics,
                    "robustness": baseline_robustness,
                },
                "config": {
                    "budget": args.al_budget,
                    "iterations": args.al_iterations,
                    "pool_size": args.al_pool_size,
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                    "seed": args.seed,
                },
            },
            "al_comparison_results.json",
            f"{args.output_dir}/logs"
        )

        # Print comparison table
        print_section("AL STRATEGY COMPARISON (Table 5)")
        print(f"{'Method':<20} {'Clean Acc':>12} {'Robust Acc':>12} {'ASR':>10} "
              f"{'Samples':>10} {'Efficiency':>12}")
        print("-" * 80)
        print(f"{'No Adv. Train':<20} "
              f"{baseline_robustness.get('clean_accuracy', 0):>12.3f} "
              f"{baseline_robustness.get('robust_accuracy', 0):>12.3f} "
              f"{baseline_robustness.get('attack_success_rate', 0):>10.3f} "
              f"{'0':>10} {'N/A':>12}")

        for strategy_name, metrics in strategy_comparison.items():
            label = f"AL-{strategy_name.capitalize()}" if strategy_name != "random" else "Random"
            print(f"{label:<20} "
                  f"{metrics.get('final_clean_accuracy', 0):>12.3f} "
                  f"{metrics.get('final_robust_accuracy', 0):>12.3f} "
                  f"{metrics.get('final_asr', 0):>10.3f} "
                  f"{metrics.get('total_labeled', 0):>10} "
                  f"{metrics.get('label_efficiency', 0)*1000:>11.3f}x")

        return all_strategy_results

    else:
        # =====================================================================
        # Single AL strategy run
        # =====================================================================
        framework = AgenticDefenseFramework(
            classifier=classifier,
            id2label=id2label,
            budget_per_iteration=args.al_budget,
            total_iterations=args.al_iterations,
            selection_strategy=args.al_strategy,
            pool_size=args.al_pool_size,
            confidence_threshold=args.confidence_threshold,
            retrain_lr=args.al_retrain_lr,
            retrain_epochs=args.al_retrain_epochs,
            regularization_lambda=args.al_reg_lambda,
            epsilon=args.epsilon,
            eval_attack_samples=args.attack_samples,
            output_dir=args.output_dir,
            seed=args.seed,
            max_length=args.max_length,
        )

        results = framework.run(
            test_df=test_df,
            train_loader=train_loader,
            dev_loader=dev_loader,
            class_weights=class_weights,
        )

        # Generate visualizations
        print_section("GENERATING VISUALIZATIONS")

        timeline = results["robustness_timeline"]

        viz.plot_al_iteration_progress(
            timeline,
            title=f"AL-{args.al_strategy.capitalize()}: Accuracy Across Iterations",
            save_name="al_iteration_progress"
        )

        viz.plot_asr_over_iterations(
            timeline,
            title=f"AL-{args.al_strategy.capitalize()}: ASR Over Iterations",
            save_name="al_asr_progress"
        )

        # Baseline vs AL comparison
        if timeline:
            final_metrics = timeline[-1]
            al_robustness = {
                "clean_accuracy": final_metrics.get("clean_accuracy", 0),
                "robust_accuracy": final_metrics.get("robust_accuracy", 0),
                "attack_success_rate": final_metrics.get("attack_success_rate", 0),
            }
            viz.plot_robustness_comparison(
                baseline_robustness, al_robustness,
                title=f"Baseline vs AL-{args.al_strategy.capitalize()}",
                save_name="al_vs_baseline_robustness"
            )

        # Label efficiency
        eff = compute_label_efficiency(timeline)

        # Save results
        save_results(
            {
                "mode": "agentic",
                "strategy": args.al_strategy,
                "results": results,
                "label_efficiency": eff,
                "baseline": {
                    "test_metrics": baseline_test_metrics,
                    "robustness": baseline_robustness,
                },
            },
            "al_results.json",
            f"{args.output_dir}/logs"
        )

        # Print summary
        print_section("AL FRAMEWORK RESULTS")
        print(f"  Strategy:          AL-{args.al_strategy.capitalize()}")
        print(f"  Total labeled:     {results['config']['total_labeled']}")
        if timeline:
            print(f"  Final Clean Acc:   {timeline[-1].get('clean_accuracy', 0):.4f}")
            print(f"  Final Robust Acc:  {timeline[-1].get('robust_accuracy', 0):.4f}")
            print(f"  Final ASR:         {timeline[-1].get('attack_success_rate', 0):.4f}")
        print(f"  Label Efficiency:  {eff.get('label_efficiency', 0)*1000:.3f}x10^-3")
        print(f"  Total Delta Robust: {eff.get('total_delta_robust', 0):+.4f}")

        return results


# ===========================================================================
# Standard Adversarial Training Pipeline (legacy)
# ===========================================================================

def run_standard_mode(args, classifier, train_df, dev_df, test_df,
                      train_loader, dev_loader, test_loader,
                      label2id, id2label, class_weights, baseline_robustness,
                      baseline_test_metrics, baseline_history, y_true, y_pred_baseline):
    """Run standard adversarial training pipeline."""

    viz = Visualizer(save_dir=f"{args.output_dir}/figures")
    label_names = [id2label[i] for i in range(len(id2label))]
    method_results = {}

    methods_to_train = ["fgsm", "pgd", "trades"] if args.method == "all" else [args.method]

    for method in methods_to_train:
        classifier.load(f"{args.output_dir}/models/baseline_model.pt")
        result = train_and_evaluate_method(
            method, classifier, train_loader, dev_loader, test_loader,
            test_df, args, class_weights, id2label, viz
        )
        method_results[method] = result

    # Multi-Attack Evaluation
    print_section("Multi-Attack Evaluation")
    adv_evaluator = AdversarialEvaluator(
        classifier=classifier, attack=CombinedAttack(seed=args.seed), id2label=id2label
    )
    attack_suite = {
        "Synonym": SynonymAttack(seed=args.seed),
        "CharSwap": CharacterSwapAttack(seed=args.seed),
        "Homoglyph": HomoglyphAttack(seed=args.seed),
        "KeyboardTypo": KeyboardTypoAttack(seed=args.seed),
        "Combined": CombinedAttack(seed=args.seed)
    }
    attack_comparison = adv_evaluator.compare_attacks(
        test_df, attacks=attack_suite,
        n_samples=min(100, args.attack_samples), random_state=args.seed
    )

    # Visualizations
    print_section("Generating Visualizations")
    viz.plot_training_history(baseline_history, "Baseline Training", save_name="01_baseline_training")
    for i, (method, result) in enumerate(method_results.items(), start=2):
        viz.plot_training_history(result["history"], f"{method.upper()} Training",
                                  save_name=f"0{i}_{method}_training")
    viz.plot_label_distribution(train_df, save_name="label_distribution")

    if len(method_results) == 1:
        method = list(method_results.keys())[0]
        viz.plot_confusion_matrix_comparison(
            y_true, y_pred_baseline, method_results[method]["y_pred"], label_names,
            save_name="confusion_matrices"
        )

    if method_results:
        best_method = list(method_results.keys())[-1]
        viz.plot_robustness_comparison(
            baseline_robustness, method_results[best_method]["robustness"],
            save_name="robustness_comparison"
        )
    viz.plot_attack_comparison(attack_comparison, save_name="attack_comparison")

    # Save results
    results = {
        "config": {
            "mode": "standard", "method": args.method,
            "model_name": args.model_name, "epochs": args.epochs,
            "adv_epochs": args.adv_epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "epsilon": args.epsilon,
            "seed": args.seed,
        },
        "dataset": {
            "train_samples": len(train_df), "dev_samples": len(dev_df),
            "test_samples": len(test_df), "num_classes": len(label2id)
        },
        "baseline": {
            "test_metrics": baseline_test_metrics, "robustness": baseline_robustness,
            "training_history": baseline_history
        },
        "methods": {},
        "attack_comparison": attack_comparison
    }
    for method, result in method_results.items():
        results["methods"][method] = {
            "test_metrics": result["test_metrics"],
            "robustness": result["robustness"],
            "training_history": result["history"]
        }
    save_results(results, "experiment_results.json", f"{args.output_dir}/logs")

    # Print summary
    print_section("EXPERIMENT SUMMARY")
    print(f"\n    BASELINE: Acc={baseline_test_metrics['accuracy']:.4f}, "
          f"F1={baseline_test_metrics['f1_macro']:.4f}, "
          f"ASR={baseline_robustness['attack_success_rate']:.4f}")

    for method, result in method_results.items():
        rob = result["robustness"]
        asr_reduction = 0
        if baseline_robustness['attack_success_rate'] > 0:
            asr_reduction = (
                (baseline_robustness['attack_success_rate'] - rob['attack_success_rate'])
                / baseline_robustness['attack_success_rate'] * 100
            )
        print(f"    {method.upper()}: Acc={result['test_metrics']['accuracy']:.4f}, "
              f"Robust={rob['robust_accuracy']:.4f}, ASR={rob['attack_success_rate']:.4f} "
              f"(ASR reduction: {asr_reduction:.1f}%)")

    return method_results


# ===========================================================================
# Model Architecture Comparison (DistilBERT vs SecBERT)
# ===========================================================================

def run_model_comparison(args):
    """
    Run full pipeline with both DistilBERT and SecBERT, then compare results.

    For each model:
    1. Train baseline
    2. Evaluate clean accuracy & F1
    3. Evaluate adversarial robustness
    4. Optionally run agentic AL framework

    Then generate comparison visualizations and summary table.
    """
    print_section("MODEL ARCHITECTURE COMPARISON: DistilBERT vs SecBERT")

    set_seed(args.seed)
    device = get_device()

    for subdir in ["models", "figures", "logs"]:
        os.makedirs(f"{args.output_dir}/{subdir}", exist_ok=True)

    # Preprocess data once (tokenization happens per model)
    print_section("STEP 1: Data Preprocessing")
    train_df, dev_df, test_df, label2id, id2label = load_and_preprocess(args)

    viz = Visualizer(save_dir=f"{args.output_dir}/figures")
    all_model_results = {}
    all_model_al_timelines = {}

    for model_key, model_hf_name in MODELS_TO_COMPARE.items():
        display_name = "DistilBERT" if "distilbert" in model_key else "SecBERT"
        print(f"\n{'#'*60}")
        print(f"  MODEL: {display_name} ({model_hf_name})")
        print(f"{'#'*60}")

        # Initialize model
        set_seed(args.seed)
        classifier = CTIClassifier(
            model_name=model_hf_name,
            num_labels=len(label2id),
            device=device
        )

        param_counts = classifier.count_parameters()
        print(f"  Parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")

        # Create dataloaders with this model's tokenizer
        train_loader, dev_loader, test_loader = create_dataloaders(
            train_df, dev_df, test_df,
            tokenizer=classifier.get_tokenizer(),
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        class_weights = compute_class_weights(
            train_df["label_id_encoded"].values,
            num_classes=len(label2id),
            device=device
        )

        # Baseline training
        print_section(f"{display_name}: Baseline Training")
        baseline_history = train_baseline(classifier, train_loader, dev_loader, args, class_weights)

        model_path = f"{args.output_dir}/models/{model_key}_baseline_model.pt"
        classifier.save(model_path)

        # Evaluate baseline
        evaluator = ModelEvaluator(classifier, id2label)
        test_metrics = evaluator.evaluate(test_loader)
        print(f"  {display_name} Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  {display_name} Test F1 (macro): {test_metrics['f1_macro']:.4f}")

        # Adversarial robustness evaluation
        print_section(f"{display_name}: Adversarial Evaluation")
        adv_evaluator = AdversarialEvaluator(
            classifier=classifier, attack=CombinedAttack(seed=args.seed), id2label=id2label
        )
        with Timer(f"{display_name} adversarial evaluation"):
            attack_results = adv_evaluator.evaluate_text_attacks(
                test_df, n_samples=args.attack_samples, random_state=args.seed
            )
        robustness = compute_attack_metrics(attack_results)

        print(f"  {display_name} Clean Accuracy:      {robustness['clean_accuracy']:.4f}")
        print(f"  {display_name} Robust Accuracy:     {robustness['robust_accuracy']:.4f}")
        print(f"  {display_name} ASR:                 {robustness['attack_success_rate']:.4f}")

        # Store results
        all_model_results[display_name] = {
            "clean_accuracy": robustness["clean_accuracy"],
            "robust_accuracy": robustness["robust_accuracy"],
            "attack_success_rate": robustness["attack_success_rate"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics.get("f1_weighted", 0),
            "accuracy": test_metrics["accuracy"],
            "total_params": f"{param_counts['total']:,}",
            "trainable_params": param_counts["trainable"],
            "baseline_history": baseline_history,
        }

        # Run agentic AL if in agentic mode
        if args.mode == "agentic":
            print_section(f"{display_name}: Agentic AL Framework")

            framework = AgenticDefenseFramework(
                classifier=classifier,
                id2label=id2label,
                budget_per_iteration=args.al_budget,
                total_iterations=args.al_iterations,
                selection_strategy=args.al_strategy,
                pool_size=args.al_pool_size,
                confidence_threshold=args.confidence_threshold,
                retrain_lr=args.al_retrain_lr,
                retrain_epochs=args.al_retrain_epochs,
                regularization_lambda=args.al_reg_lambda,
                epsilon=args.epsilon,
                eval_attack_samples=args.attack_samples,
                output_dir=args.output_dir,
                seed=args.seed,
                max_length=args.max_length,
            )

            al_results = framework.run(
                test_df=test_df,
                train_loader=train_loader,
                dev_loader=dev_loader,
                class_weights=class_weights,
            )

            al_timeline = al_results["robustness_timeline"]
            all_model_al_timelines[display_name] = {args.al_strategy: al_timeline}

            # Update results with post-AL metrics
            if al_timeline:
                final = al_timeline[-1]
                all_model_results[display_name]["al_robust_accuracy"] = final.get("robust_accuracy", 0)
                all_model_results[display_name]["al_asr"] = final.get("attack_success_rate", 0)
                all_model_results[display_name]["al_clean_accuracy"] = final.get("clean_accuracy", 0)

    # =========================================================================
    # Generate Comparison Visualizations
    # =========================================================================
    print_section("MODEL COMPARISON VISUALIZATIONS")

    viz.plot_model_comparison(
        all_model_results,
        title="Model Architecture Comparison: DistilBERT vs SecBERT",
        save_name="model_comparison"
    )

    if all_model_al_timelines:
        viz.plot_model_al_comparison(
            all_model_al_timelines,
            title="Model Comparison: AL Robustness Progression",
            save_name="model_al_comparison"
        )

    # Comparison metrics
    arch_comparison = compare_model_architectures(all_model_results)

    # Save results
    save_results(
        {
            "mode": "model_comparison",
            "models": all_model_results,
            "comparison": arch_comparison,
            "config": {
                "al_mode": args.mode,
                "al_budget": args.al_budget,
                "al_iterations": args.al_iterations,
                "al_strategy": args.al_strategy,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "seed": args.seed,
            },
        },
        "model_comparison_results.json",
        f"{args.output_dir}/logs"
    )

    # Print comparison table
    print_section("MODEL COMPARISON RESULTS")
    print(f"{'Model':<15} {'Clean Acc':>12} {'Robust Acc':>12} {'ASR':>10} "
          f"{'F1 Macro':>10} {'Params':>15}")
    print("-" * 80)
    for model_name, metrics in all_model_results.items():
        print(f"{model_name:<15} "
              f"{metrics.get('clean_accuracy', 0):>12.4f} "
              f"{metrics.get('robust_accuracy', 0):>12.4f} "
              f"{metrics.get('attack_success_rate', 0):>10.4f} "
              f"{metrics.get('f1_macro', 0):>10.4f} "
              f"{metrics.get('total_params', 'N/A'):>15}")

    if all_model_al_timelines:
        print(f"\n{'Model':<15} {'AL Robust Acc':>15} {'AL ASR':>10} {'AL Clean Acc':>15}")
        print("-" * 60)
        for model_name, metrics in all_model_results.items():
            if "al_robust_accuracy" in metrics:
                print(f"{model_name:<15} "
                      f"{metrics['al_robust_accuracy']:>15.4f} "
                      f"{metrics['al_asr']:>10.4f} "
                      f"{metrics['al_clean_accuracy']:>15.4f}")

    if "best_per_metric" in arch_comparison:
        print("\nBest model per metric:")
        for metric, best_model in arch_comparison["best_per_metric"].items():
            print(f"  {metric:<25} -> {best_model}")

    return all_model_results


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    """Main training and evaluation pipeline."""
    args = parse_args()

    # Setup
    print_section("ADVERSARIAL ROBUSTNESS FOR CTI CLASSIFICATION")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == "agentic":
        print(f"Strategy: AL-{args.al_strategy.capitalize()}")
        print(f"Budget: {args.al_budget}, Iterations: {args.al_iterations}")
    else:
        print(f"Method: {args.method.upper()}")

    # =========================================================================
    # Model Comparison Mode (runs both models end-to-end)
    # =========================================================================
    if args.model_compare:
        run_model_comparison(args)
        print_section("COMPLETE")
        return

    set_seed(args.seed)
    device = get_device()

    # Create output directories
    for subdir in ["models", "figures", "logs"]:
        os.makedirs(f"{args.output_dir}/{subdir}", exist_ok=True)

    # =========================================================================
    # STEP 1: Data Preprocessing
    # =========================================================================
    print_section("STEP 1: Data Preprocessing")
    train_df, dev_df, test_df, label2id, id2label = load_and_preprocess(args)

    # =========================================================================
    # STEP 2: Initialize Model and DataLoaders
    # =========================================================================
    print_section("STEP 2: Initialize Model and DataLoaders")

    classifier = CTIClassifier(
        model_name=args.model_name,
        num_labels=len(label2id),
        device=device
    )

    train_loader, dev_loader, test_loader = create_dataloaders(
        train_df, dev_df, test_df,
        tokenizer=classifier.get_tokenizer(),
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    class_weights = compute_class_weights(
        train_df["label_id_encoded"].values,
        num_classes=len(label2id),
        device=device
    )

    # =========================================================================
    # STEP 3: Baseline Training
    # =========================================================================
    print_section("STEP 3: Baseline Training")
    baseline_history = train_baseline(classifier, train_loader, dev_loader, args, class_weights)

    # Evaluate baseline
    print("\nEvaluating baseline on test set...")
    evaluator = ModelEvaluator(classifier, id2label)
    baseline_test_metrics = evaluator.evaluate(test_loader)
    print(f"Baseline Test Accuracy: {baseline_test_metrics['accuracy']:.4f}")
    print(f"Baseline Test F1 (macro): {baseline_test_metrics['f1_macro']:.4f}")
    y_true, y_pred_baseline = evaluator.evaluate_loader(test_loader, "Baseline predictions")

    # Baseline adversarial evaluation
    print_section("STEP 4: Baseline Adversarial Evaluation")
    adv_evaluator = AdversarialEvaluator(
        classifier=classifier, attack=CombinedAttack(seed=args.seed), id2label=id2label
    )
    with Timer("Baseline adversarial evaluation"):
        baseline_attack_results = adv_evaluator.evaluate_text_attacks(
            test_df, n_samples=args.attack_samples, random_state=args.seed
        )
    baseline_robustness = compute_attack_metrics(baseline_attack_results)
    print(f"\nBaseline Robustness:")
    print(f"  Clean Accuracy:      {baseline_robustness['clean_accuracy']:.4f}")
    print(f"  Robust Accuracy:     {baseline_robustness['robust_accuracy']:.4f}")
    print(f"  Attack Success Rate: {baseline_robustness['attack_success_rate']:.4f}")

    # =========================================================================
    # STEP 5: Mode-specific pipeline
    # =========================================================================
    if args.mode == "agentic":
        run_agentic_mode(
            args, classifier, train_df, dev_df, test_df,
            train_loader, dev_loader, test_loader,
            label2id, id2label, class_weights, baseline_robustness,
            baseline_test_metrics, baseline_history, y_true, y_pred_baseline
        )
    else:
        run_standard_mode(
            args, classifier, train_df, dev_df, test_df,
            train_loader, dev_loader, test_loader,
            label2id, id2label, class_weights, baseline_robustness,
            baseline_test_metrics, baseline_history, y_true, y_pred_baseline
        )

    # =========================================================================
    # Done
    # =========================================================================
    print(f"\n    OUTPUT FILES")
    print(f"    {'='*40}")
    print(f"    Models:  {args.output_dir}/models/")
    print(f"    Figures: {args.output_dir}/figures/")
    print(f"    Logs:    {args.output_dir}/logs/")
    print_section("COMPLETE")


if __name__ == "__main__":
    main()
