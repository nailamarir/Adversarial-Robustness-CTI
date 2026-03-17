"""
Visualization Module
Comprehensive plotting functions for CTI classification and robustness analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os


class Visualizer:
    """Main visualization class with consistent styling"""

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        save_dir: str = "./outputs/figures",
        color_palette: str = "husl"
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = save_dir
        self.color_palette = color_palette

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn" in plt.style.available else "default")

        # Color schemes
        self.colors = {
            "baseline": "#3498db",
            "adversarial": "#2ecc71",
            "attack": "#e74c3c",
            "neutral": "#95a5a6",
            "primary": "#2c3e50",
            "secondary": "#9b59b6"
        }

    def save_figure(self, fig: plt.Figure, name: str) -> str:
        """Save figure to file"""
        path = os.path.join(self.save_dir, f"{name}.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
        return path

    def plot_training_history(
        self,
        history: Dict,
        title: str = "Training History",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot training loss and validation metrics over epochs"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history.get("train_loss", [])) + 1)

        # Loss plot
        ax1 = axes[0]
        if "train_loss" in history:
            ax1.plot(epochs, history["train_loss"], 'b-o', label="Train Loss", linewidth=2, markersize=8)
        if "clean_loss" in history:
            ax1.plot(epochs, history["clean_loss"], 'g--s', label="Clean Loss", linewidth=2, markersize=6)
        if "adv_loss" in history:
            ax1.plot(epochs, history["adv_loss"], 'r--^', label="Adv Loss", linewidth=2, markersize=6)

        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Metrics plot
        ax2 = axes[1]
        if "dev_acc" in history:
            ax2.plot(epochs, history["dev_acc"], 'b-o', label="Dev Accuracy", linewidth=2, markersize=8)
        if "dev_f1" in history:
            ax2.plot(epochs, history["dev_f1"], 'g-s', label="Dev F1 (Macro)", linewidth=2, markersize=8)

        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Score", fontsize=12)
        ax2.set_title("Validation Metrics", fontsize=14, fontweight="bold")
        ax2.legend(loc="best", fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
            fmt = ".2f"
        else:
            fmt = "d"

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels or range(cm.shape[1]),
            yticklabels=labels or range(cm.shape[0]),
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_confusion_matrix_comparison(
        self,
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_adversarial: np.ndarray,
        labels: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot side-by-side confusion matrices"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        for ax, y_pred, title, cmap in [
            (axes[0], y_pred_baseline, "Baseline Model", "Blues"),
            (axes[1], y_pred_adversarial, "Adversarially Trained Model", "Greens")
        ]:
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                xticklabels=labels or range(cm.shape[1]),
                yticklabels=labels or range(cm.shape[0]),
                ax=ax,
                cbar_kws={"shrink": 0.8}
            )

            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle("Confusion Matrix Comparison", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_label_distribution(
        self,
        df: pd.DataFrame,
        label_column: str = "label_mapped",
        title: str = "Label Distribution",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot label distribution bar chart"""
        fig, ax = plt.subplots(figsize=(12, 6))

        label_counts = df[label_column].value_counts().sort_values(ascending=True)

        colors = sns.color_palette(self.color_palette, len(label_counts))

        bars = ax.barh(range(len(label_counts)), label_counts.values, color=colors)
        ax.set_yticks(range(len(label_counts)))
        ax.set_yticklabels(label_counts.index)

        # Add count labels
        for bar, count in zip(bars, label_counts.values):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'{count}', va='center', fontsize=10)

        ax.set_xlabel("Number of Samples", fontsize=12)
        ax.set_ylabel("Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_robustness_comparison(
        self,
        baseline_metrics: Dict,
        adversarial_metrics: Dict,
        title: str = "Robustness Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of baseline vs adversarial model robustness"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Accuracy comparison
        ax1 = axes[0]
        metrics = ["clean_accuracy", "robust_accuracy"]
        x = np.arange(len(metrics))
        width = 0.35

        baseline_vals = [baseline_metrics.get(m, 0) for m in metrics]
        adv_vals = [adversarial_metrics.get(m, 0) for m in metrics]

        bars1 = ax1.bar(x - width/2, baseline_vals, width, label="Baseline",
                        color=self.colors["baseline"], edgecolor="black")
        bars2 = ax1.bar(x + width/2, adv_vals, width, label="Adversarial",
                        color=self.colors["adversarial"], edgecolor="black")

        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["Clean", "Robust"])
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

        # ASR comparison
        ax2 = axes[1]
        asr_baseline = baseline_metrics.get("attack_success_rate", 0)
        asr_adv = adversarial_metrics.get("attack_success_rate", 0)

        bars = ax2.bar(["Baseline", "Adversarial"], [asr_baseline, asr_adv],
                       color=[self.colors["baseline"], self.colors["adversarial"]],
                       edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight="bold")

        ax2.set_ylabel("Attack Success Rate", fontsize=12)
        ax2.set_title("ASR Comparison (Lower is Better)", fontsize=14, fontweight="bold")
        ax2.set_ylim(0, max(asr_baseline, asr_adv) * 1.3 + 0.05)

        # Improvement summary
        ax3 = axes[2]
        ax3.axis('off')

        # Calculate improvements
        asr_reduction = 0
        if asr_baseline > 0:
            asr_reduction = (asr_baseline - asr_adv) / asr_baseline * 100

        robust_improvement = adversarial_metrics.get("robust_accuracy", 0) - baseline_metrics.get("robust_accuracy", 0)

        summary_text = f"""
        IMPROVEMENT SUMMARY
        ─────────────────────

        ASR Reduction:     {asr_reduction:.1f}%

        Robust Acc Change: {robust_improvement:+.4f}

        Baseline ASR:      {asr_baseline:.4f}
        Adversarial ASR:   {asr_adv:.4f}

        Baseline Clean:    {baseline_metrics.get('clean_accuracy', 0):.4f}
        Adversarial Clean: {adversarial_metrics.get('clean_accuracy', 0):.4f}
        """

        ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_attack_comparison(
        self,
        attack_results: Dict[str, Dict],
        title: str = "Attack Method Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of different attack methods"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        attack_names = list(attack_results.keys())
        n_attacks = len(attack_names)

        colors = sns.color_palette(self.color_palette, n_attacks)

        # ASR comparison
        ax1 = axes[0]
        asrs = [attack_results[name].get("attack_success_rate", 0) for name in attack_names]
        bars = ax1.bar(attack_names, asrs, color=colors, edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        ax1.set_ylabel("Attack Success Rate", fontsize=12)
        ax1.set_title("ASR by Attack Type", fontsize=14, fontweight="bold")
        ax1.set_ylim(0, max(asrs) * 1.3 + 0.05 if asrs else 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Accuracy comparison
        ax2 = axes[1]
        x = np.arange(n_attacks)
        width = 0.35

        clean_accs = [attack_results[name].get("clean_accuracy", 0) for name in attack_names]
        robust_accs = [attack_results[name].get("robust_accuracy", 0) for name in attack_names]

        ax2.bar(x - width/2, clean_accs, width, label="Clean Accuracy", color=self.colors["baseline"])
        ax2.bar(x + width/2, robust_accs, width, label="Robust Accuracy", color=self.colors["adversarial"])

        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Clean vs Robust Accuracy", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(attack_names)
        ax2.legend()
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_per_class_metrics(
        self,
        metrics: Dict,
        metric_name: str = "f1-score",
        title: str = "Per-Class Performance",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot per-class metrics (from classification report)"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract per-class metrics (skip avg rows)
        class_metrics = {k: v for k, v in metrics.items()
                        if isinstance(v, dict) and k not in
                        ["accuracy", "macro avg", "weighted avg"]}

        if not class_metrics:
            ax.text(0.5, 0.5, "No per-class metrics available",
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        labels = list(class_metrics.keys())
        values = [class_metrics[label].get(metric_name, 0) for label in labels]

        # Sort by value
        sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_pairs)

        colors = sns.color_palette("RdYlGn", len(labels))
        colors = [colors[int(v * (len(colors) - 1))] for v in values]

        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="black")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)

        ax.set_xlabel(metric_name.title(), fontsize=12)
        ax.set_ylabel("Class", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.1)
        ax.axvline(x=np.mean(values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(values):.3f}')
        ax.legend()

        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_confidence_distribution(
        self,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        title: str = "Prediction Confidence Distribution",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of prediction confidences"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Get max probability (confidence) for each prediction
        confidences = np.max(probabilities, axis=1)
        correct = predictions == true_labels

        # Overall confidence distribution
        ax1 = axes[0]
        ax1.hist(confidences, bins=50, color=self.colors["primary"],
                edgecolor="black", alpha=0.7)
        ax1.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.set_xlabel("Confidence", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Overall Confidence Distribution", fontsize=14, fontweight="bold")
        ax1.legend()

        # Confidence by correctness
        ax2 = axes[1]
        ax2.hist(confidences[correct], bins=30, alpha=0.7,
                label=f'Correct (n={np.sum(correct)})',
                color=self.colors["adversarial"], edgecolor="black")
        ax2.hist(confidences[~correct], bins=30, alpha=0.7,
                label=f'Incorrect (n={np.sum(~correct)})',
                color=self.colors["attack"], edgecolor="black")
        ax2.set_xlabel("Confidence", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Confidence by Correctness", fontsize=14, fontweight="bold")
        ax2.legend()

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_robustness_by_class(
        self,
        per_class_results: Dict,
        title: str = "Robustness by Class",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot per-class robustness metrics"""
        fig, ax = plt.subplots(figsize=(12, 8))

        classes = list(per_class_results.keys())
        asrs = [per_class_results[c].get("asr", 0) for c in classes]

        # Sort by ASR
        sorted_pairs = sorted(zip(classes, asrs), key=lambda x: x[1], reverse=True)
        classes, asrs = zip(*sorted_pairs)

        colors = sns.color_palette("RdYlGn_r", len(classes))

        bars = ax.barh(range(len(classes)), asrs, color=colors, edgecolor="black")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)

        # Add value labels
        for bar, val in zip(bars, asrs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)

        ax.set_xlabel("Attack Success Rate", fontsize=12)
        ax.set_ylabel("Class", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axvline(x=np.mean(asrs), color='blue', linestyle='--',
                   label=f'Mean ASR: {np.mean(asrs):.3f}')
        ax.legend()

        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def plot_al_learning_curves(
        self,
        strategy_timelines: Dict[str, List[Dict]],
        title: str = "Active Learning: Robust Accuracy vs Labeled Samples",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot robust accuracy as a function of labeled samples for different
        selection strategies (Figure 3 in the paper).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"entropy": "#2ecc71", "random": "#95a5a6", "margin": "#e67e22"}
        markers = {"entropy": "o", "random": "s", "margin": "^"}

        for strategy, timeline in strategy_timelines.items():
            n_labeled = [entry.get("n_labeled_samples", 0) for entry in timeline]
            robust_acc = [entry.get("robust_accuracy", 0) for entry in timeline]
            color = colors.get(strategy, "#3498db")
            marker = markers.get(strategy, "D")
            label = f"AL-{strategy.capitalize()}" if strategy != "random" else "Random"
            ax.plot(n_labeled, robust_acc, f'-{marker}', label=label,
                    color=color, linewidth=2, markersize=8)

        ax.set_xlabel("Number of Labeled Samples", fontsize=12)
        ax.set_ylabel("Robust Accuracy", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)
        return fig

    def plot_al_iteration_progress(
        self,
        robustness_timeline: List[Dict],
        title: str = "Robust Accuracy Across AL Iterations",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot robust accuracy progression across active learning iterations
        (Figure 4 in the paper).
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        iterations = [entry.get("iteration", i) for i, entry in enumerate(robustness_timeline)]
        robust_acc = [entry.get("robust_accuracy", 0) for entry in robustness_timeline]

        ax.plot(iterations, robust_acc, '-o', color=self.colors["adversarial"],
                linewidth=2.5, markersize=10, label="Robust Accuracy")

        # Add clean accuracy line
        clean_acc = [entry.get("clean_accuracy", 0) for entry in robustness_timeline]
        ax.plot(iterations, clean_acc, '--s', color=self.colors["baseline"],
                linewidth=2, markersize=8, alpha=0.7, label="Clean Accuracy")

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)
        return fig

    def plot_al_strategy_comparison(
        self,
        comparison: Dict[str, Dict],
        title: str = "Active Learning Strategy Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Bar chart comparing strategies on robust accuracy, ASR, and label efficiency
        (Table 5 / Table 6 in the paper).
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        strategies = list(comparison.keys())
        n = len(strategies)
        colors = sns.color_palette(self.color_palette, n)
        display_names = []
        for s in strategies:
            display_names.append(f"AL-{s.capitalize()}" if s != "random" else "Random")

        # Robust Accuracy
        ax = axes[0]
        vals = [comparison[s].get("final_robust_accuracy", 0) for s in strategies]
        bars = ax.bar(display_names, vals, color=colors, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax.set_ylabel("Robust Accuracy", fontsize=12)
        ax.set_title("Robust Accuracy", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)

        # ASR (lower is better)
        ax = axes[1]
        vals = [comparison[s].get("final_asr", 0) for s in strategies]
        bars = ax.bar(display_names, vals, color=colors, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax.set_ylabel("Attack Success Rate", fontsize=12)
        ax.set_title("ASR (Lower is Better)", fontsize=14, fontweight="bold")

        # Label Efficiency
        ax = axes[2]
        vals = [comparison[s].get("label_efficiency", 0) * 1000 for s in strategies]
        bars = ax.bar(display_names, vals, color=colors, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax.set_ylabel("Label Efficiency (x1000)", fontsize=12)
        ax.set_title("Label Efficiency", fontsize=14, fontweight="bold")

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)
        return fig

    def plot_asr_over_iterations(
        self,
        robustness_timeline: List[Dict],
        title: str = "Attack Success Rate Over Iterations",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot ASR reduction across active learning iterations."""
        fig, ax = plt.subplots(figsize=(8, 6))

        iterations = [entry.get("iteration", i) for i, entry in enumerate(robustness_timeline)]
        asr = [entry.get("attack_success_rate", 0) for entry in robustness_timeline]

        ax.plot(iterations, asr, '-o', color=self.colors["attack"],
                linewidth=2.5, markersize=10)
        ax.fill_between(iterations, asr, alpha=0.2, color=self.colors["attack"])

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Attack Success Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)
        return fig

    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict],
        title: str = "Model Architecture Comparison: DistilBERT vs SecBERT",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of different model architectures (e.g., DistilBERT vs SecBERT)
        showing clean accuracy, robust accuracy, ASR, and F1 score side by side.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        model_names = list(model_results.keys())
        n = len(model_names)
        colors = sns.color_palette(self.color_palette, n)

        # --- Panel 1: Clean & Robust Accuracy ---
        ax = axes[0, 0]
        x = np.arange(n)
        width = 0.35
        clean_vals = [model_results[m].get("clean_accuracy", 0) for m in model_names]
        robust_vals = [model_results[m].get("robust_accuracy", 0) for m in model_names]
        bars1 = ax.bar(x - width/2, clean_vals, width, label="Clean Acc",
                       color=self.colors["baseline"], edgecolor="black")
        bars2 = ax.bar(x + width/2, robust_vals, width, label="Robust Acc",
                       color=self.colors["adversarial"], edgecolor="black")
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Clean vs Robust Accuracy", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()

        # --- Panel 2: ASR ---
        ax = axes[0, 1]
        asr_vals = [model_results[m].get("attack_success_rate", 0) for m in model_names]
        bars = ax.bar(model_names, asr_vals, color=colors, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12,
                        fontweight="bold")
        ax.set_ylabel("Attack Success Rate", fontsize=12)
        ax.set_title("ASR (Lower is Better)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(asr_vals) * 1.3 + 0.05 if asr_vals else 1)

        # --- Panel 3: F1 Scores ---
        ax = axes[1, 0]
        f1_macro = [model_results[m].get("f1_macro", 0) for m in model_names]
        f1_weighted = [model_results[m].get("f1_weighted", 0) for m in model_names]
        bars1 = ax.bar(x - width/2, f1_macro, width, label="F1 Macro",
                       color=self.colors["primary"], edgecolor="black")
        bars2 = ax.bar(x + width/2, f1_weighted, width, label="F1 Weighted",
                       color=self.colors["secondary"], edgecolor="black")
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title("F1 Score Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()

        # --- Panel 4: Summary Table ---
        ax = axes[1, 1]
        ax.axis('off')
        summary_lines = ["MODEL COMPARISON SUMMARY", "=" * 40]
        for m in model_names:
            r = model_results[m]
            summary_lines.append(f"\n{m}")
            summary_lines.append(f"  Clean Accuracy:      {r.get('clean_accuracy', 0):.4f}")
            summary_lines.append(f"  Robust Accuracy:     {r.get('robust_accuracy', 0):.4f}")
            summary_lines.append(f"  ASR:                 {r.get('attack_success_rate', 0):.4f}")
            summary_lines.append(f"  F1 Macro:            {r.get('f1_macro', 0):.4f}")
            summary_lines.append(f"  Parameters:          {r.get('total_params', 'N/A')}")
        summary_text = "\n".join(summary_lines)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)
        return fig

    def plot_model_al_comparison(
        self,
        model_al_results: Dict[str, Dict[str, List[Dict]]],
        title: str = "Model Comparison: AL Robustness Progression",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot AL robustness timeline for multiple models on the same chart.

        Args:
            model_al_results: {model_name: {strategy: timeline_list}}
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        model_colors = {"DistilBERT": "#3498db", "SecBERT": "#e74c3c"}
        model_markers = {"DistilBERT": "o", "SecBERT": "s"}

        # Robust accuracy over labeled samples
        ax = axes[0]
        for model_name, strategy_timelines in model_al_results.items():
            # Use entropy strategy if available, else first available
            strategy = "entropy" if "entropy" in strategy_timelines else list(strategy_timelines.keys())[0]
            timeline = strategy_timelines[strategy]
            n_labeled = [e.get("n_labeled_samples", 0) for e in timeline]
            robust_acc = [e.get("robust_accuracy", 0) for e in timeline]
            color = model_colors.get(model_name, "#95a5a6")
            marker = model_markers.get(model_name, "D")
            ax.plot(n_labeled, robust_acc, f'-{marker}', label=model_name,
                    color=color, linewidth=2, markersize=8)
        ax.set_xlabel("Number of Labeled Samples", fontsize=12)
        ax.set_ylabel("Robust Accuracy", fontsize=12)
        ax.set_title("Robust Accuracy vs Labeled Samples", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # ASR over iterations
        ax = axes[1]
        for model_name, strategy_timelines in model_al_results.items():
            strategy = "entropy" if "entropy" in strategy_timelines else list(strategy_timelines.keys())[0]
            timeline = strategy_timelines[strategy]
            iterations = [e.get("iteration", i) for i, e in enumerate(timeline)]
            asr = [e.get("attack_success_rate", 0) for e in timeline]
            color = model_colors.get(model_name, "#95a5a6")
            marker = model_markers.get(model_name, "D")
            ax.plot(iterations, asr, f'-{marker}', label=model_name,
                    color=color, linewidth=2, markersize=8)
        ax.set_xlabel("AL Iteration", fontsize=12)
        ax.set_ylabel("Attack Success Rate", fontsize=12)
        ax.set_title("ASR Over AL Iterations", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)
        return fig

    def plot_text_length_analysis(
        self,
        df: pd.DataFrame,
        text_column: str = "full_text",
        label_column: str = "label_mapped",
        title: str = "Text Length Analysis",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Analyze and plot text length distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Calculate text lengths
        lengths = df[text_column].str.len()

        # Overall distribution
        ax1 = axes[0]
        ax1.hist(lengths, bins=50, color=self.colors["primary"],
                edgecolor="black", alpha=0.7)
        ax1.axvline(np.mean(lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(lengths):.0f}')
        ax1.axvline(np.median(lengths), color='green', linestyle='--',
                   label=f'Median: {np.median(lengths):.0f}')
        ax1.set_xlabel("Text Length (characters)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Overall Text Length Distribution", fontsize=14, fontweight="bold")
        ax1.legend()

        # By class (boxplot)
        ax2 = axes[1]
        df_plot = df.copy()
        df_plot["text_length"] = lengths

        # Get top classes
        top_classes = df[label_column].value_counts().head(10).index.tolist()
        df_plot = df_plot[df_plot[label_column].isin(top_classes)]

        sns.boxplot(data=df_plot, x=label_column, y="text_length", ax=ax2,
                   palette=self.color_palette)
        ax2.set_xlabel("Label", fontsize=12)
        ax2.set_ylabel("Text Length", fontsize=12)
        ax2.set_title("Text Length by Class (Top 10)", fontsize=14, fontweight="bold")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        return fig


# Convenience functions
def plot_training_history(history: Dict, save_path: Optional[str] = None) -> plt.Figure:
    viz = Visualizer()
    return viz.plot_training_history(history, save_name=save_path)


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None) -> plt.Figure:
    viz = Visualizer()
    return viz.plot_confusion_matrix(y_true, y_pred, labels, save_name=save_path)


def plot_label_distribution(df, label_column="label_mapped", save_path=None) -> plt.Figure:
    viz = Visualizer()
    return viz.plot_label_distribution(df, label_column, save_name=save_path)


def plot_robustness_comparison(baseline_metrics, adversarial_metrics, save_path=None) -> plt.Figure:
    viz = Visualizer()
    return viz.plot_robustness_comparison(baseline_metrics, adversarial_metrics, save_name=save_path)


def plot_attack_comparison(attack_results, save_path=None) -> plt.Figure:
    viz = Visualizer()
    return viz.plot_attack_comparison(attack_results, save_name=save_path)


def plot_per_class_metrics(metrics, metric_name="f1-score", save_path=None) -> plt.Figure:
    viz = Visualizer()
    return viz.plot_per_class_metrics(metrics, metric_name, save_name=save_path)


def plot_roc_curves(y_true, y_prob, labels=None, save_path=None) -> plt.Figure:
    """Plot ROC curves for multi-class classification"""
    viz = Visualizer()
    fig, ax = plt.subplots(figsize=(10, 8))

    n_classes = y_prob.shape[1]
    y_bin = label_binarize(y_true, classes=range(n_classes))

    colors = sns.color_palette(viz.color_palette, n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        label_name = labels[i] if labels else f"Class {i}"
        ax.plot(fpr, tpr, color=colors[i], lw=2,
               label=f'{label_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    if save_path:
        viz.save_figure(fig, save_path)

    return fig


def plot_confidence_distribution(probs, preds, true_labels, save_path=None) -> plt.Figure:
    viz = Visualizer()
    return viz.plot_confidence_distribution(probs, preds, true_labels, save_name=save_path)


def create_full_report(
    baseline_history: Dict,
    adversarial_history: Dict,
    baseline_metrics: Dict,
    adversarial_metrics: Dict,
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_adversarial: np.ndarray,
    labels: List[str],
    attack_comparison: Optional[Dict] = None,
    save_dir: str = "./outputs/figures"
) -> List[str]:
    """Generate full visualization report"""
    viz = Visualizer(save_dir=save_dir)
    saved_files = []

    print("Generating visualization report...")

    # 1. Training histories
    fig = viz.plot_training_history(baseline_history, "Baseline Training",
                                    save_name="01_baseline_training")
    saved_files.append("01_baseline_training.png")
    plt.close(fig)

    fig = viz.plot_training_history(adversarial_history, "Adversarial Training",
                                    save_name="02_adversarial_training")
    saved_files.append("02_adversarial_training.png")
    plt.close(fig)

    # 2. Confusion matrices
    fig = viz.plot_confusion_matrix_comparison(
        y_true, y_pred_baseline, y_pred_adversarial, labels,
        save_name="03_confusion_matrices"
    )
    saved_files.append("03_confusion_matrices.png")
    plt.close(fig)

    # 3. Robustness comparison
    fig = viz.plot_robustness_comparison(
        baseline_metrics, adversarial_metrics,
        save_name="04_robustness_comparison"
    )
    saved_files.append("04_robustness_comparison.png")
    plt.close(fig)

    # 4. Attack comparison (if available)
    if attack_comparison:
        fig = viz.plot_attack_comparison(attack_comparison,
                                         save_name="05_attack_comparison")
        saved_files.append("05_attack_comparison.png")
        plt.close(fig)

    print(f"Generated {len(saved_files)} visualization files in {save_dir}")
    return saved_files
