"""
Selection Agent
================
Role: Strategist
Perception: Candidate pool U from the Detection Agent; model confidence scores P(y|x).
Action: Compute prediction entropy H(x) for each candidate; select top-B samples S*.
Goal: Maximize robustness gain per labeled sample within the annotation budget.

The Selection Agent represents the core methodological contribution of this work.
Rather than treating all adversarial candidates equally, this agent intelligently
prioritizes samples that provide maximum learning value for robustness improvement.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import euclidean_distances

from ..models.classifier import CTIClassifier
from .detection_agent import AdversarialCandidate


@dataclass
class ScoredCandidate:
    """An adversarial candidate scored by uncertainty and utility."""
    candidate: AdversarialCandidate
    entropy: float
    margin: float  # Difference between top-2 predicted class probabilities
    utility: float = 0.0  # Adversarial Sample Utility (ASU)
    loss: float = 0.0  # Cross-entropy loss on true label
    confidence_flip: float = 0.0  # 1 - P(true_class)
    rank: int = 0


class SelectionAgent:
    """
    Selection Agent: Scores candidates by prediction uncertainty and selects
    the most informative samples S* within the labeling budget B.

    Implements two selection strategies:
    1. Entropy-based (AL-Entropy): Selects highest-entropy samples
    2. Margin-based (AL-Margin): Selects smallest-margin samples

    The entropy-based strategy is the primary contribution, as described in
    Algorithm 1 of the paper.
    """

    def __init__(
        self,
        classifier: CTIClassifier,
        budget_per_iteration: int = 50,
        strategy: str = "entropy",
        max_length: int = 512,
        beta: float = 0.5,
    ):
        self.classifier = classifier
        self.budget = budget_per_iteration
        self.strategy = strategy
        self.max_length = max_length
        self.device = classifier.device
        self.beta = beta  # Weight for margin in composite acquisition

        # Selection history
        self.selection_history: List[Dict] = []

    def compute_entropy(self, probs: np.ndarray) -> float:
        """
        Compute prediction entropy: H(x) = -sum_{c=1}^{C} P(y=c|x) * log P(y=c|x)

        H(x) ~ 0: Model is highly confident (easy sample, low informativeness)
        H(x) ~ log(C): Maximum uncertainty (hard sample, high informativeness)
        """
        probs_clipped = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs_clipped * np.log(probs_clipped)))

    def compute_margin(self, probs: np.ndarray) -> float:
        """
        Compute margin: difference between top-2 predicted class probabilities.

        Small margin = high uncertainty (similar to two top classes).
        """
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) >= 2:
            return float(sorted_probs[0] - sorted_probs[1])
        return float(sorted_probs[0])

    def compute_utility(self, loss: float, confidence_flip: float, margin: float) -> float:
        """
        Compute Adversarial Sample Utility (ASU) — Eq. 8 in paper.

        U(x_adv) = α·loss + β·(1 - P(y_true|x_adv)) + γ·(1 - margin)

        High utility = sample that maximally shifts the decision boundary:
        - High loss: model is wrong and penalized
        - High confidence flip: model is confident in wrong class
        - Low margin: near decision boundary between top classes
        """
        alpha, gamma = 0.4, 0.2  # beta is self.beta for margin weight
        return alpha * loss + self.beta * confidence_flip + gamma * (1.0 - margin)

    def score_candidates(
        self,
        candidate_pool: List[AdversarialCandidate]
    ) -> List[ScoredCandidate]:
        """
        Score all candidates in pool U by adversarial utility.

        For each adversarial candidate x in U:
        1. Compute softmax probabilities p = softmax(M_theta(x))
        2. Compute entropy H(x) = -sum(p_c * log(p_c))
        3. Compute margin = P(y1|x) - P(y2|x)
        4. Compute loss = -log(P(y_true|x))
        5. Compute utility U(x) = α·loss + β·confidence_flip + γ·(1-margin)

        Returns:
            List of ScoredCandidate objects with entropy, margin, and utility scores.
        """
        scored: List[ScoredCandidate] = []
        self.classifier.eval_mode()

        for candidate in candidate_pool:
            probs = self.classifier.get_probabilities(
                candidate.text, max_length=self.max_length
            )
            probs_np = probs.cpu().numpy()

            entropy = self.compute_entropy(probs_np)
            margin = self.compute_margin(probs_np)

            # Compute loss and confidence flip for true label
            true_label = candidate.true_label
            if 0 <= true_label < len(probs_np):
                p_true = float(np.clip(probs_np[true_label], 1e-10, 1.0))
                loss = -np.log(p_true)
                confidence_flip = 1.0 - p_true
            else:
                loss = 5.0  # max loss
                confidence_flip = 1.0

            utility = self.compute_utility(loss, confidence_flip, margin)

            scored.append(ScoredCandidate(
                candidate=candidate,
                entropy=entropy,
                margin=margin,
                utility=utility,
                loss=loss,
                confidence_flip=confidence_flip,
            ))

        return scored

    def select_top_b(
        self,
        scored_candidates: List[ScoredCandidate],
        budget: Optional[int] = None
    ) -> List[ScoredCandidate]:
        """
        Select top-B most informative samples S* from scored pool.

        S* = argmax_{S subset U, |S|=B} sum_{x in S} H(x)   (for entropy strategy)
        S* = argmin_{S subset U, |S|=B} margin(x)            (for margin strategy)

        This implements the budget-constrained selection (Eq. 8 in the paper).

        Args:
            scored_candidates: List of scored candidates
            budget: Number of samples to select (defaults to self.budget)

        Returns:
            Selected top-B candidates with ranks assigned
        """
        B = budget or self.budget

        if self.strategy == "entropy":
            # Select highest entropy (most uncertain)
            sorted_candidates = sorted(
                scored_candidates,
                key=lambda x: x.entropy,
                reverse=True  # Highest entropy first
            )
        elif self.strategy == "margin":
            # Select smallest margin (most uncertain between top-2 classes)
            sorted_candidates = sorted(
                scored_candidates,
                key=lambda x: x.margin,
                reverse=False  # Smallest margin first
            )
        elif self.strategy == "composite":
            # Adversarial Sample Utility — boundary-shifting score
            # U(x) = α·loss + β·confidence_flip + γ·(1-margin)
            sorted_candidates = sorted(
                scored_candidates,
                key=lambda x: x.utility,
                reverse=True  # Highest utility first
            )
        elif self.strategy == "coreset":
            # Core-Set: greedy k-center in probability space
            sorted_candidates = self._coreset_selection(scored_candidates, B)
        elif self.strategy == "entropy_coreset":
            # Entropy + Core-Set: top-2B by entropy, then filter to B via k-center
            entropy_sorted = sorted(
                scored_candidates,
                key=lambda x: x.entropy,
                reverse=True
            )
            top_2b = entropy_sorted[:min(2 * B, len(entropy_sorted))]
            sorted_candidates = self._coreset_selection(top_2b, B)
        elif self.strategy == "random":
            # Random baseline for comparison
            sorted_candidates = list(scored_candidates)
            np.random.shuffle(sorted_candidates)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Select top-B
        selected = sorted_candidates[:B]

        # Assign ranks
        for i, sc in enumerate(selected):
            sc.rank = i + 1

        # Log selection
        if selected:
            entropies = [sc.entropy for sc in selected]
            self.selection_history.append({
                "budget": B,
                "pool_size": len(scored_candidates),
                "selected": len(selected),
                "strategy": self.strategy,
                "avg_entropy": float(np.mean(entropies)),
                "max_entropy": float(np.max(entropies)),
                "min_entropy": float(np.min(entropies)),
            })

        return selected

    def select_from_pool(
        self,
        candidate_pool: List[AdversarialCandidate],
        budget: Optional[int] = None
    ) -> List[ScoredCandidate]:
        """
        Full selection pipeline: score + select top-B.

        This is the main entry point implementing Steps 1-2 of Algorithm 1.
        """
        B = budget or self.budget

        if not candidate_pool:
            print("Selection Agent: Empty candidate pool, nothing to select.")
            return []

        # Step 1: Score all candidates by uncertainty
        scored = self.score_candidates(candidate_pool)

        # Step 2: Select top-B uncertain samples
        selected = self.select_top_b(scored, B)

        print(f"Selection Agent: Scored {len(scored)} candidates, "
              f"selected top-{len(selected)} by {self.strategy}")

        if selected:
            avg_entropy = np.mean([s.entropy for s in selected])
            print(f"  Average entropy of selected: {avg_entropy:.4f}")

        return selected

    def get_selection_history(self) -> List[Dict]:
        """Return selection history across iterations."""
        return self.selection_history.copy()

    def _coreset_selection(
        self,
        scored_candidates: List[ScoredCandidate],
        budget: int
    ) -> List[ScoredCandidate]:
        """
        Greedy k-center selection in probability space (Core-Set).
        Maximizes diversity by iteratively picking the candidate farthest
        from all previously selected candidates.
        """
        if len(scored_candidates) <= budget:
            return list(scored_candidates)

        # Get probability vectors for all candidates
        prob_vectors = []
        for sc in scored_candidates:
            probs = self.classifier.get_probabilities(
                sc.candidate.text, max_length=self.max_length
            )
            prob_vectors.append(probs.cpu().numpy())
        prob_matrix = np.array(prob_vectors)

        # Greedy k-center
        selected_indices = [0]  # Start with first candidate
        for _ in range(budget - 1):
            dists = euclidean_distances(prob_matrix, prob_matrix[selected_indices])
            min_dists = dists.min(axis=1)  # Distance to nearest selected
            min_dists[selected_indices] = -1  # Exclude already selected
            next_idx = int(np.argmax(min_dists))
            selected_indices.append(next_idx)

        return [scored_candidates[i] for i in selected_indices]

    def get_entropy_statistics(
        self,
        scored_candidates: List[ScoredCandidate]
    ) -> Dict:
        """Compute statistics about the entropy distribution of candidates."""
        if not scored_candidates:
            return {}

        entropies = [sc.entropy for sc in scored_candidates]
        margins = [sc.margin for sc in scored_candidates]

        return {
            "n_candidates": len(scored_candidates),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "entropy_median": float(np.median(entropies)),
            "entropy_min": float(np.min(entropies)),
            "entropy_max": float(np.max(entropies)),
            "margin_mean": float(np.mean(margins)),
            "margin_std": float(np.std(margins)),
        }
