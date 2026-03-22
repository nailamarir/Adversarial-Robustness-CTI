"""
Configuration file for Adversarial Robustness CTI Project
Contains all hyperparameters, paths, and settings
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class PathConfig:
    """Data and output paths configuration"""
    # Data paths (modify based on your environment)
    base_data_dir: str = "/content/drive/MyDrive/AnnoCTR"  # Google Colab
    # base_data_dir: str = "./data/AnnoCTR"  # Local

    # Output paths
    output_dir: str = "./outputs"
    model_dir: str = "./outputs/models"
    figures_dir: str = "./outputs/figures"
    logs_dir: str = "./outputs/logs"

    @property
    def text_dir(self) -> str:
        return f"{self.base_data_dir}/all/text"

    @property
    def labels_dir(self) -> str:
        return f"{self.base_data_dir}/linking_mitre_only"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_name: str = "distilbert-base-uncased"
    max_seq_length: int = 256  # Paper Table 4
    num_labels: int = 16  # Top 15 + OTHER
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Baseline training (Paper Table 4)
    batch_size: int = 16
    num_epochs: int = 5               # Paper Table 4: 5 initial epochs
    learning_rate: float = 2e-5       # Paper Table 4: 2e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # FGSM adversarial training
    fgsm_epochs: int = 3  # More adversarial epochs
    fgsm_learning_rate: float = 1e-5
    fgsm_epsilon: float = 0.01  # Stronger perturbation
    clean_loss_weight: float = 0.7
    adv_loss_weight: float = 0.3

    # PGD adversarial training (NEW)
    pgd_epochs: int = 3
    pgd_epsilon: float = 0.01
    pgd_alpha: float = 0.003  # Step size
    pgd_steps: int = 7  # Number of PGD steps
    pgd_learning_rate: float = 1e-5

    # TRADES training (NEW)
    trades_epochs: int = 3
    trades_beta: float = 6.0  # Trade-off parameter (higher = more robust)
    trades_epsilon: float = 0.01
    trades_step_size: float = 0.003
    trades_num_steps: int = 7
    trades_learning_rate: float = 1e-5

    # Early stopping
    patience: int = 5  # More patience

    # Random seed
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration"""
    top_k_labels: int = 15
    other_sample_size: int = 600  # Balance OTHER class (original method)
    test_val_size: float = 0.2
    random_state: int = 42

    # Enhanced preprocessing options
    use_enhanced_preprocessing: bool = True
    min_samples_per_class: int = 50  # Oversample minority classes to this
    max_samples_per_class: int = 800  # Undersample majority classes to this
    clean_text: bool = True  # Apply CTI-specific text cleaning
    augment_data: bool = True  # Enable data augmentation
    augmentation_factor: float = 0.3  # Augmentation intensity
    truncation_strategy: str = "head_tail"  # head, tail, head_tail, important


@dataclass
class ActiveLearningConfig:
    """Active Learning hyperparameters (from Table 4 of the paper)"""
    budget_per_iteration: int = 50          # B: samples selected per iteration
    total_iterations: int = 5               # T: number of AL iterations
    selection_strategy: str = "entropy"     # entropy, margin, or random
    pool_size: int = 500                    # Size of adversarial candidate pool U
    confidence_threshold: float = 0.7       # Threshold for detection agent
    retrain_lr: float = 2e-5               # Learning rate for fine-tuning
    retrain_epochs: int = 2                 # Epochs per AL iteration
    regularization_lambda: float = 0.01     # L_reg weight (Eq. 9)
    clean_mix_ratio: float = 0.3            # Ratio of clean samples in mixed training
    eval_attack_samples: int = 200          # Samples for robustness evaluation


@dataclass
class AttackConfig:
    """Adversarial attack configuration"""
    num_synonym_replacements: int = 3
    num_char_swaps: int = 2
    eval_subset_size: int = 200
    fgsm_eval_epsilon: float = 0.1


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    figure_size: tuple = (10, 6)
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "husl"
    save_format: str = "png"

    # Color schemes
    baseline_color: str = "#3498db"
    adversarial_color: str = "#2ecc71"
    attack_color: str = "#e74c3c"
    neutral_color: str = "#95a5a6"


@dataclass
class Config:
    """Main configuration class combining all configs"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Validate and setup after initialization"""
        import os
        os.makedirs(self.paths.output_dir, exist_ok=True)
        os.makedirs(self.paths.model_dir, exist_ok=True)
        os.makedirs(self.paths.figures_dir, exist_ok=True)
        os.makedirs(self.paths.logs_dir, exist_ok=True)


def get_config() -> Config:
    """Factory function to get configuration"""
    return Config()


# Models available for comparison
MODELS_TO_COMPARE = {
    "distilbert": "distilbert-base-uncased",
    "secbert": "jackaduma/SecBERT",
    "secroberta": "jackaduma/SecRoBERTa",
}

# All selection strategies for paper experiments
SELECTION_STRATEGIES = [
    "entropy",        # Pure Entropy (φ_H)
    "margin",         # Pure Margin (φ_M)
    "composite",      # Composite (φ_H + β·φ_M) — A3L
    "coreset",        # Core-Set (greedy k-center)
    "entropy_coreset", # Entropy + Core-Set hybrid
    "random",         # Random baseline
]


# Label mapping (MITRE ATT&CK technique IDs)
MITRE_LABEL_NAMES = {
    "1049": "Exfiltration Over Web Service",
    "1170": "Mshta",
    "1348": "Serverless Execution",
    "1401": "SMS Control",
    "161": "Command & Scripting Interpreter",
    "195": "Firmware Corruption",
    "202": "Exploitation of Remote Services",
    "368": "Permission Groups Discovery",
    "452": "Hook",
    "467": "Dynamic API Resolution",
    "514": "Elevated Execution with Prompt",
    "544": "Scheduled Task/Job",
    "581": "Masquerading",
    "685": "Email Hiding Rules",
    "688": "Remote System Discovery",
    "OTHER": "Other Techniques"
}
