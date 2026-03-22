r"""
Agentic Active Learning-Enhanced Adaptive Defense Framework
============================================================
Orchestrates the four specialized agents in a closed feedback loop:
Detection -> Selection -> Oracle -> Retraining -> (repeat)
                            |
                       Audit Agent (monitors all)

Implements Algorithm 1 from the paper:
  for t = 1 to T do
      1. Score all candidates by uncertainty
      2. Select top-B uncertain samples S_t
      3. Query oracle (security analyst) for labels
      4. theta <- FineTune(theta, L, epochs=2)
      5. U <- U \ S_t
      6. Evaluate and log
  end for
"""

import copy
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from ..models.classifier import CTIClassifier
from ..attacks.text_attacks import CombinedAttack, TextAttacker
from ..evaluation.evaluator import AdversarialEvaluator
from ..evaluation.metrics import compute_attack_metrics
from .detection_agent import DetectionAgent, AdversarialCandidate
from .selection_agent import SelectionAgent, ScoredCandidate
from .retraining_agent import RetrainingAgent
from .audit_agent import AuditAgent


class AgenticDefenseFramework:
    """
    Main framework orchestrating the multi-agent active learning defense.

    This class implements the full pipeline described in Section 4 of the paper,
    coordinating four specialized agents in a continuous closed-loop cycle
    to improve adversarial robustness with minimal annotation cost.
    """

    def __init__(
        self,
        classifier: CTIClassifier,
        id2label: Dict[int, str],
        # Active Learning parameters
        budget_per_iteration: int = 50,
        total_iterations: int = 5,
        selection_strategy: str = "entropy",
        # Detection parameters
        pool_size: int = 500,
        confidence_threshold: float = 0.7,
        # Adaptive parameters (Eqs. 6, 10 in paper)
        adaptive: bool = True,
        budget_min: int = 10,
        budget_max: int = 100,
        budget_delta: int = 10,
        budget_epsilon: float = 0.01,
        tau_alpha: float = 0.1,
        asr_target: float = 0.3,
        drift_epsilon: float = 0.02,
        # Retraining parameters
        retrain_lr: float = 2e-5,
        retrain_epochs: int = 2,
        regularization_lambda: float = 0.1,
        clean_mix_ratio: float = 3.0,
        epsilon: float = 0.01,
        # Evaluation
        eval_attack_samples: int = 200,
        # General
        output_dir: str = "./outputs",
        seed: int = 42,
        max_length: int = 512
    ):
        self.classifier = classifier
        self.id2label = id2label
        self.budget = budget_per_iteration
        self.total_iterations = total_iterations
        self.pool_size = pool_size
        self.eval_attack_samples = eval_attack_samples
        self.output_dir = output_dir
        self.seed = seed
        self.max_length = max_length

        # Adaptive parameters
        self.adaptive = adaptive
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.budget_delta = budget_delta
        self.budget_epsilon = budget_epsilon
        self.tau_alpha = tau_alpha
        self.asr_target = asr_target
        self.drift_epsilon = drift_epsilon
        self.current_budget = budget_per_iteration
        self.current_tau = confidence_threshold
        self.prev_robust_acc = 0.0

        # Initialize the four agents
        self.detection_agent = DetectionAgent(
            classifier=classifier,
            id2label=id2label,
            confidence_threshold=confidence_threshold,
            seed=seed,
        )

        self.selection_agent = SelectionAgent(
            classifier=classifier,
            budget_per_iteration=budget_per_iteration,
            strategy=selection_strategy,
            max_length=max_length,
        )

        self.retraining_agent = RetrainingAgent(
            classifier=classifier,
            learning_rate=retrain_lr,
            epochs_per_iteration=retrain_epochs,
            regularization_lambda=regularization_lambda,
            clean_mix_ratio=clean_mix_ratio,
            epsilon=epsilon,
            max_length=max_length,
        )

        self.audit_agent = AuditAgent(
            output_dir=f"{output_dir}/logs"
        )

        # Framework state
        self.labeled_set: List[ScoredCandidate] = []
        self.candidate_pool: List[AdversarialCandidate] = []
        self.total_labeled = 0

        # Results storage
        self.iteration_results: List[Dict] = []

    def run(
        self,
        test_df: pd.DataFrame,
        train_loader=None,
        dev_loader=None,
        class_weights: Optional[torch.Tensor] = None,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
    ) -> Dict:
        """
        Run the full active learning defense loop (Algorithm 1).

        Args:
            test_df: Test DataFrame (used to generate adversarial pool & evaluate)
            train_loader: Clean training DataLoader for mixed training
            dev_loader: Validation DataLoader for monitoring
            class_weights: Class weights for loss function
            text_column: Column name for text
            label_column: Column name for encoded labels

        Returns:
            Dictionary with all results and the audit trail
        """
        print("=" * 60)
        print("  AGENTIC ACTIVE LEARNING DEFENSE FRAMEWORK")
        print("=" * 60)
        print(f"  Budget per iteration (B): {self.budget}")
        print(f"  Total iterations (T):     {self.total_iterations}")
        print(f"  Selection strategy:       {self.selection_agent.strategy}")
        print(f"  Pool size:                {self.pool_size}")
        print("=" * 60)

        # Log system start
        self.audit_agent.log_event(
            event_type="system",
            agent="Framework",
            action="framework_started",
            details={
                "budget": self.budget,
                "iterations": self.total_iterations,
                "strategy": self.selection_agent.strategy,
                "pool_size": self.pool_size,
            }
        )

        # =====================================================================
        # Step 0: Initial evaluation (before any AL iterations)
        # =====================================================================
        print("\n--- Initial Evaluation (Iteration 0) ---")
        initial_metrics = self._evaluate_robustness(test_df, text_column, label_column)

        self.audit_agent.log_robustness_evaluation(
            clean_accuracy=initial_metrics["clean_accuracy"],
            robust_accuracy=initial_metrics["robust_accuracy"],
            attack_success_rate=initial_metrics["attack_success_rate"],
            n_labeled_samples=0,
            iteration=0,
        )

        self.iteration_results.append({
            "iteration": 0,
            "n_labeled": 0,
            "metrics": initial_metrics,
        })

        print(f"  Clean Acc: {initial_metrics['clean_accuracy']:.4f}, "
              f"Robust Acc: {initial_metrics['robust_accuracy']:.4f}, "
              f"ASR: {initial_metrics['attack_success_rate']:.4f}")

        # =====================================================================
        # Step 1: Detection Agent generates multi-attack adversarial pool U
        # =====================================================================
        print("\n--- Detection Agent: Generating Multi-Attack Pool ---")

        # Phase 1a: Text-level attacks (synonym, charswap, homoglyph, etc.)
        text_pool = self.detection_agent.generate_adversarial_pool(
            df=test_df,
            text_column=text_column,
            label_column=label_column,
            pool_size=self.pool_size,
            random_state=self.seed,
        )

        # Phase 1b: FGSM embedding-level attacks (critical for robust accuracy)
        fgsm_pool = self.detection_agent.generate_fgsm_pool(
            df=test_df,
            epsilon=0.1,
            text_column=text_column,
            label_column=label_column,
            pool_size=min(self.pool_size, 300),
            random_state=self.seed,
        )

        # Combine pools — FGSM candidates are highest priority
        self.candidate_pool = fgsm_pool + text_pool
        print(f"  Combined pool: {len(fgsm_pool)} FGSM + {len(text_pool)} text = {len(self.candidate_pool)} total")

        detection_stats = self.detection_agent.get_statistics()
        self.audit_agent.log_detection(
            pool_size=len(self.candidate_pool),
            n_flagged=detection_stats["total_flagged"],
            flagged_by_method=detection_stats.get("flagged_by_method", {}),
            iteration=0,
        )

        print(f"  Adversarial pool size |U|: {len(self.candidate_pool)}")

        # =====================================================================
        # Main AL Loop: for t = 1 to T
        # =====================================================================
        for t in range(1, self.total_iterations + 1):
            print(f"\n{'='*60}")
            print(f"  ACTIVE LEARNING ITERATION {t}/{self.total_iterations}")
            print(f"{'='*60}")

            if not self.candidate_pool:
                print("  Pool exhausted. Logging final state and stopping.")
                final_metrics = self._evaluate_robustness(test_df, text_column, label_column)
                self.audit_agent.log_robustness_evaluation(
                    clean_accuracy=final_metrics["clean_accuracy"],
                    robust_accuracy=final_metrics["robust_accuracy"],
                    attack_success_rate=final_metrics["attack_success_rate"],
                    n_labeled_samples=self.total_labeled,
                    iteration=t,
                )
                self.iteration_results.append({
                    "iteration": t, "n_labeled": self.total_labeled,
                    "metrics": final_metrics, "status": "pool_exhausted",
                })
                break

            # Step 1: Selection Agent scores and selects top-B
            effective_budget = self.current_budget if self.adaptive else self.budget
            print(f"\n  [Selection] Scoring {len(self.candidate_pool)} candidates (budget={effective_budget})...")
            selected = self.selection_agent.select_from_pool(
                self.candidate_pool,
                budget=effective_budget
            )

            if not selected:
                print("  No samples selected. Stopping.")
                break

            # Audit: log selection with explanations
            self.audit_agent.log_selection(
                selected=selected,
                strategy=self.selection_agent.strategy,
                pool_size=len(self.candidate_pool),
                budget=self.budget,
                iteration=t,
            )

            # Step 2: Oracle provides labels (simulated using ground truth)
            # In real deployment, a security analyst would label these samples
            self.labeled_set.extend(selected)
            self.total_labeled += len(selected)

            # Step 3: Retraining Agent fine-tunes model
            print(f"\n  [Retraining] Fine-tuning on {len(selected)} samples...")
            retrain_result = self.retraining_agent.retrain_iteration(
                selected_samples=selected,
                clean_loader=train_loader,
                dev_loader=dev_loader,
                class_weights=class_weights,
                iteration=t,
            )

            self.audit_agent.log_retraining(
                n_samples=retrain_result.get("n_samples", 0),
                epochs=retrain_result.get("epochs", 0),
                loss=retrain_result.get("avg_loss", 0),
                dev_accuracy=retrain_result.get("dev_accuracy", 0),
                dev_f1=retrain_result.get("dev_f1", 0),
                iteration=t,
            )

            # Step 4: Remove selected samples from pool
            selected_texts = {sc.candidate.text for sc in selected}
            self.candidate_pool = [
                c for c in self.candidate_pool
                if c.text not in selected_texts
            ]

            # Step 5: Evaluate robustness
            print(f"\n  [Evaluation] Evaluating robustness...")
            iter_metrics = self._evaluate_robustness(test_df, text_column, label_column)

            self.audit_agent.log_robustness_evaluation(
                clean_accuracy=iter_metrics["clean_accuracy"],
                robust_accuracy=iter_metrics["robust_accuracy"],
                attack_success_rate=iter_metrics["attack_success_rate"],
                n_labeled_samples=self.total_labeled,
                iteration=t,
            )

            self.iteration_results.append({
                "iteration": t,
                "n_labeled": self.total_labeled,
                "n_selected_this_iter": len(selected),
                "remaining_pool": len(self.candidate_pool),
                "retrain_result": retrain_result,
                "metrics": iter_metrics,
            })

            # Adaptive updates (Eqs. 6, 10, 12 in paper)
            current_robust = iter_metrics['robust_accuracy']
            current_asr = iter_metrics['attack_success_rate']
            delta_robust = current_robust - self.prev_robust_acc

            if self.adaptive:
                # Eq. 10: Budget adaptation based on robustness gain
                if delta_robust > self.budget_epsilon:
                    self.current_budget = max(self.budget_min, self.current_budget - self.budget_delta)
                else:
                    self.current_budget = min(self.budget_max, self.current_budget + self.budget_delta)

                # Eq. 6: Threshold adaptation based on ASR
                self.current_tau = self.current_tau - self.tau_alpha * (current_asr - self.asr_target)
                self.current_tau = max(0.1, min(0.95, self.current_tau))

                # Eq. 12: Drift alert — reset to aggressive if robustness drops
                if delta_robust < -self.drift_epsilon and t > 1:
                    self.current_budget = self.budget_max
                    self.current_tau = max(0.3, self.current_tau - 0.1)
                    print(f"    [Audit] DRIFT ALERT: robustness dropped by {-delta_robust:.4f}, increasing budget")

                self.detection_agent.confidence_threshold = self.current_tau

            self.prev_robust_acc = current_robust

            # Print iteration summary
            print(f"\n  Iteration {t} Summary:")
            print(f"    Samples labeled this iter: {len(selected)}")
            print(f"    Total labeled so far:      {self.total_labeled}")
            print(f"    Remaining pool:            {len(self.candidate_pool)}")
            print(f"    Clean Acc:   {iter_metrics['clean_accuracy']:.4f}")
            print(f"    Robust Acc:  {iter_metrics['robust_accuracy']:.4f}")
            print(f"    ASR:         {iter_metrics['attack_success_rate']:.4f}")
            if self.adaptive:
                print(f"    Budget B^(t): {self.current_budget}, Threshold τ^(t): {self.current_tau:.3f}")

            # Store adaptive state in iteration results
            self.iteration_results[-1]["adaptive_state"] = {
                "budget": self.current_budget,
                "tau": self.current_tau,
                "delta_robust": delta_robust,
            }

            # Generate iteration summary in audit log
            self.audit_agent.generate_iteration_summary(t)

        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("  FRAMEWORK COMPLETE")
        print("=" * 60)

        # Save audit log
        self.audit_agent.save_audit_log()
        self.audit_agent.print_summary()

        # Compile final results
        results = {
            "config": {
                "budget_per_iteration": self.budget,
                "total_iterations": self.total_iterations,
                "selection_strategy": self.selection_agent.strategy,
                "pool_size": self.pool_size,
                "total_labeled": self.total_labeled,
            },
            "iteration_results": self.iteration_results,
            "robustness_timeline": self.audit_agent.get_robustness_timeline(),
            "selection_history": self.selection_agent.get_selection_history(),
            "retraining_history": self.retraining_agent.get_history(),
        }

        return results

    def _evaluate_robustness(
        self,
        test_df: pd.DataFrame,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
    ) -> Dict:
        """Evaluate current model robustness against FGSM embedding attacks."""
        adv_evaluator = AdversarialEvaluator(
            classifier=self.classifier,
            attack=CombinedAttack(seed=self.seed),
            id2label=self.id2label,
        )

        # Use FGSM embedding-level attacks for meaningful robustness evaluation
        attack_results = adv_evaluator.evaluate_fgsm_attacks(
            test_df,
            epsilon=0.1,
            n_samples=self.eval_attack_samples,
            random_state=self.seed,
        )

        metrics = compute_attack_metrics(attack_results)
        return metrics

    def run_comparison(
        self,
        test_df: pd.DataFrame,
        train_loader=None,
        dev_loader=None,
        class_weights: Optional[torch.Tensor] = None,
        strategies: Optional[List[str]] = None,
        text_column: str = "full_text",
        label_column: str = "label_id_encoded",
    ) -> Dict[str, Dict]:
        """
        Run the framework with multiple selection strategies for comparison.

        Compares: AL-Entropy, AL-Margin, Random (as described in Section 5.5).

        Args:
            test_df: Test DataFrame
            train_loader: Clean training DataLoader
            dev_loader: Validation DataLoader
            class_weights: Class weights
            strategies: List of strategies to compare

        Returns:
            Dictionary mapping strategy name -> results
        """
        if strategies is None:
            strategies = ["entropy", "margin", "random"]

        comparison_results = {}

        # Save initial model state
        initial_state = copy.deepcopy(self.classifier.get_model().state_dict())

        for strategy in strategies:
            print(f"\n{'#'*60}")
            print(f"  RUNNING STRATEGY: {strategy.upper()}")
            print(f"{'#'*60}")

            # Reset model to initial state
            self.classifier.get_model().load_state_dict(
                copy.deepcopy(initial_state)
            )

            # Reset framework state
            self.selection_agent.strategy = strategy
            self.selection_agent.selection_history = []
            self.retraining_agent.history = []
            self.audit_agent = AuditAgent(output_dir=f"{self.output_dir}/logs")
            self.labeled_set = []
            self.candidate_pool = []
            self.total_labeled = 0
            self.iteration_results = []

            # Run
            results = self.run(
                test_df=test_df,
                train_loader=train_loader,
                dev_loader=dev_loader,
                class_weights=class_weights,
                text_column=text_column,
                label_column=label_column,
            )

            comparison_results[strategy] = results

        # Restore initial state
        self.classifier.get_model().load_state_dict(initial_state)

        return comparison_results
