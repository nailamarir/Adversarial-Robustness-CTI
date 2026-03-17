"""
Audit Agent
=============
Role: Overseer
Perception: All agent decisions, performance metrics, and model states.
Action: Log events, generate explanations, track robustness drift.
Goal: Ensure transparency, compliance, and human oversight throughout
      the defense lifecycle.

The Audit Agent operates asynchronously, receiving signals from the Detection,
Selection, and Retraining agents and maintaining a running audit log.
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from .detection_agent import AdversarialCandidate
from .selection_agent import ScoredCandidate


@dataclass
class AuditEvent:
    """A single audit log entry."""
    timestamp: str
    event_type: str  # detection, selection, retraining, evaluation, system
    agent: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    iteration: int = 0


class AuditAgent:
    """
    Audit Agent: Monitors all agent decisions, tracks robustness metrics,
    and provides explainable decision logs for compliance and human oversight.

    Key functions:
    1. Decision Logging - All detection, selection, and retraining events
    2. Robustness Metrics Tracking - Clean accuracy, robust accuracy, ASR over iterations
    3. Explanation Generation - Why each sample was prioritized
    4. Compliance Support - Audit trail for security governance requirements
    """

    def __init__(self, output_dir: str = "./outputs/logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Audit log
        self.events: List[AuditEvent] = []

        # Robustness metrics over iterations
        self.robustness_timeline: List[Dict] = []

        # Per-iteration summaries
        self.iteration_summaries: List[Dict] = []

        # System start time
        self.start_time = datetime.now().isoformat()

    def log_event(
        self,
        event_type: str,
        agent: str,
        action: str,
        details: Optional[Dict] = None,
        iteration: int = 0
    ):
        """Log a single audit event."""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            agent=agent,
            action=action,
            details=details or {},
            iteration=iteration,
        )
        self.events.append(event)

    def log_detection(
        self,
        pool_size: int,
        n_flagged: int,
        flagged_by_method: Dict[str, int],
        iteration: int = 0
    ):
        """Log detection agent results."""
        self.log_event(
            event_type="detection",
            agent="DetectionAgent",
            action="adversarial_pool_generated",
            details={
                "pool_size": pool_size,
                "n_flagged": n_flagged,
                "flagged_by_method": flagged_by_method,
            },
            iteration=iteration,
        )

    def log_selection(
        self,
        selected: List[ScoredCandidate],
        strategy: str,
        pool_size: int,
        budget: int,
        iteration: int = 0
    ):
        """
        Log selection agent results with explanations.

        For each selected sample, records WHY it was prioritized
        (uncertainty score, feature characteristics).
        """
        selection_details = {
            "strategy": strategy,
            "pool_size": pool_size,
            "budget": budget,
            "n_selected": len(selected),
        }

        if selected:
            entropies = [sc.entropy for sc in selected]
            selection_details["entropy_stats"] = {
                "mean": float(np.mean(entropies)),
                "std": float(np.std(entropies)),
                "min": float(np.min(entropies)),
                "max": float(np.max(entropies)),
            }

            # Top-5 sample explanations
            explanations = []
            for sc in selected[:5]:
                explanations.append({
                    "rank": sc.rank,
                    "label": sc.candidate.label_name,
                    "entropy": round(sc.entropy, 4),
                    "margin": round(sc.margin, 4),
                    "detection_method": sc.candidate.detection_method,
                    "confidence": round(sc.candidate.confidence_score, 4),
                    "reason": self._generate_explanation(sc),
                })
            selection_details["top_sample_explanations"] = explanations

        self.log_event(
            event_type="selection",
            agent="SelectionAgent",
            action="samples_selected",
            details=selection_details,
            iteration=iteration,
        )

    def log_retraining(
        self,
        n_samples: int,
        epochs: int,
        loss: float,
        dev_accuracy: float,
        dev_f1: float,
        iteration: int = 0
    ):
        """Log retraining agent results."""
        self.log_event(
            event_type="retraining",
            agent="RetrainingAgent",
            action="model_fine_tuned",
            details={
                "n_samples": n_samples,
                "epochs": epochs,
                "final_loss": round(loss, 4),
                "dev_accuracy": round(dev_accuracy, 4),
                "dev_f1": round(dev_f1, 4),
            },
            iteration=iteration,
        )

    def log_robustness_evaluation(
        self,
        clean_accuracy: float,
        robust_accuracy: float,
        attack_success_rate: float,
        n_labeled_samples: int,
        iteration: int = 0
    ):
        """
        Track robustness metrics over iterations.

        This is the core tracking function for the Audit Agent,
        monitoring how the model's robustness changes as more
        actively-selected samples are incorporated.
        """
        metrics = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "clean_accuracy": round(clean_accuracy, 4),
            "robust_accuracy": round(robust_accuracy, 4),
            "attack_success_rate": round(attack_success_rate, 4),
            "n_labeled_samples": n_labeled_samples,
        }

        # Compute label efficiency (Eq. 11 in paper)
        if iteration > 0 and self.robustness_timeline:
            baseline_robust = self.robustness_timeline[0]["robust_accuracy"]
            delta_robust = robust_accuracy - baseline_robust
            if n_labeled_samples > 0:
                metrics["label_efficiency"] = round(delta_robust / n_labeled_samples, 6)
            else:
                metrics["label_efficiency"] = 0.0
            metrics["delta_robust_accuracy"] = round(delta_robust, 4)

        self.robustness_timeline.append(metrics)

        self.log_event(
            event_type="evaluation",
            agent="AuditAgent",
            action="robustness_evaluated",
            details=metrics,
            iteration=iteration,
        )

    def _generate_explanation(self, scored: ScoredCandidate) -> str:
        """
        Generate a human-readable explanation for why a sample was selected.

        This supports the Explainability requirement of the framework.
        """
        candidate = scored.candidate
        parts = []

        # Entropy-based reasoning
        if scored.entropy > 1.5:
            parts.append(f"Very high uncertainty (entropy={scored.entropy:.3f})")
        elif scored.entropy > 1.0:
            parts.append(f"High uncertainty (entropy={scored.entropy:.3f})")
        else:
            parts.append(f"Moderate uncertainty (entropy={scored.entropy:.3f})")

        # Margin-based reasoning
        if scored.margin < 0.1:
            parts.append("nearly equal top-2 class probabilities")
        elif scored.margin < 0.3:
            parts.append("small margin between top predictions")

        # Detection method
        if "misclassification" in candidate.detection_method:
            parts.append(f"attack caused misclassification via {candidate.metadata.get('attack_type', 'unknown')}")
        elif "low_confidence" in candidate.detection_method:
            parts.append(f"low model confidence ({candidate.confidence_score:.3f})")

        return "; ".join(parts)

    def generate_iteration_summary(self, iteration: int) -> Dict:
        """Generate a summary for a complete AL iteration."""
        iteration_events = [e for e in self.events if e.iteration == iteration]

        summary = {
            "iteration": iteration,
            "n_events": len(iteration_events),
            "event_types": {},
        }

        for event in iteration_events:
            etype = event.event_type
            summary["event_types"][etype] = summary["event_types"].get(etype, 0) + 1

        # Add robustness metrics if available
        for rm in self.robustness_timeline:
            if rm["iteration"] == iteration:
                summary["robustness"] = rm
                break

        self.iteration_summaries.append(summary)
        return summary

    def get_robustness_timeline(self) -> List[Dict]:
        """Return robustness metrics over all iterations."""
        return self.robustness_timeline.copy()

    def get_full_audit_log(self) -> Dict:
        """Return the complete audit log."""
        return {
            "system_start": self.start_time,
            "total_events": len(self.events),
            "events": [asdict(e) for e in self.events],
            "robustness_timeline": self.robustness_timeline,
            "iteration_summaries": self.iteration_summaries,
        }

    def save_audit_log(self, filename: str = "audit_log.json"):
        """Save the full audit log to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)

        log = self.get_full_audit_log()

        with open(filepath, 'w') as f:
            json.dump(log, f, indent=2, default=str)

        print(f"Audit log saved to: {filepath}")
        return filepath

    def print_summary(self):
        """Print a human-readable summary of the audit trail."""
        print("\n" + "=" * 60)
        print("  AUDIT AGENT SUMMARY")
        print("=" * 60)
        print(f"  Total events logged: {len(self.events)}")
        print(f"  AL iterations tracked: {len(self.robustness_timeline)}")

        if self.robustness_timeline:
            first = self.robustness_timeline[0]
            last = self.robustness_timeline[-1]

            print(f"\n  Robustness Progression:")
            print(f"    Initial  -> Clean: {first['clean_accuracy']:.4f}, "
                  f"Robust: {first['robust_accuracy']:.4f}, "
                  f"ASR: {first['attack_success_rate']:.4f}")
            print(f"    Final    -> Clean: {last['clean_accuracy']:.4f}, "
                  f"Robust: {last['robust_accuracy']:.4f}, "
                  f"ASR: {last['attack_success_rate']:.4f}")

            if 'delta_robust_accuracy' in last:
                print(f"    Delta Robust Accuracy: {last['delta_robust_accuracy']:+.4f}")
            if 'label_efficiency' in last:
                print(f"    Label Efficiency: {last['label_efficiency']:.6f}")

        print("=" * 60)
