# Adversarial Robustness for CTI NLP Models

A modular framework for training and evaluating adversarially robust text classifiers for Cyber Threat Intelligence (CTI) report classification.

## Project Overview

This project implements adversarial training techniques to improve the robustness of NLP models against adversarial attacks in the cybersecurity domain. It uses the AnnoCTR dataset to classify CTI reports into MITRE ATT&CK technique categories.

### Key Features

- **Modular Architecture**: Clean, reusable code organized into logical modules
- **Multiple Attack Types**: Synonym substitution, character swaps, homoglyphs, keyboard typos
- **FGSM Training**: Fast Gradient Sign Method for adversarial robustness
- **Comprehensive Evaluation**: Clean accuracy, robust accuracy, attack success rate
- **Rich Visualizations**: Training curves, confusion matrices, robustness comparisons

## Project Structure

```
Adversarial-Robustness-CTI/
├── main.py                     # Main training script
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── configs/
│   └── config.py               # Configuration settings
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Data loading and preprocessing
│   │   └── dataset.py          # PyTorch Dataset classes
│   ├── models/
│   │   └── classifier.py       # CTI classifier model
│   ├── training/
│   │   ├── trainer.py          # Baseline and adversarial trainers
│   │   └── losses.py           # Loss functions
│   ├── attacks/
│   │   ├── text_attacks.py     # Text-level adversarial attacks
│   │   └── fgsm.py             # FGSM attack implementation
│   ├── evaluation/
│   │   ├── evaluator.py        # Model evaluation utilities
│   │   └── metrics.py          # Evaluation metrics
│   ├── visualization/
│   │   └── plots.py            # Visualization functions
│   └── utils/
│       └── helpers.py          # Utility functions
├── outputs/
│   ├── models/                 # Saved model checkpoints
│   ├── figures/                # Generated visualizations
│   └── logs/                   # Experiment results
└── notebooks/                  # (Optional) Jupyter notebooks
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/daaalrijjal/Adversarial-Robustness-CTI.git
cd Adversarial-Robustness-CTI
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### 5. Download the Dataset

Download the AnnoCTR dataset from:
- [boschresearch/anno-ctr-lrec-coling-2024](https://github.com/boschresearch/anno-ctr-lrec-coling-2024)

## Usage

### Quick Start

```bash
python main.py --data_path /path/to/AnnoCTR --epochs 5
```

### Full Options

```bash
python main.py \
    --data_path /path/to/AnnoCTR \
    --epochs 5 \
    --fgsm_epochs 1 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --fgsm_epsilon 0.005 \
    --model_name jackaduma/SecBERT \
    --max_length 512 \
    --top_k_labels 15 \
    --attack_samples 200 \
    --output_dir ./outputs \
    --seed 42
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | (required) | Path to AnnoCTR dataset |
| `--epochs` | 5 | Baseline training epochs |
| `--fgsm_epochs` | 1 | FGSM training epochs |
| `--batch_size` | 8 | Training batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--fgsm_epsilon` | 0.005 | FGSM perturbation strength |
| `--model_name` | jackaduma/SecBERT | Pretrained model |
| `--max_length` | 512 | Max sequence length |
| `--top_k_labels` | 15 | Number of label classes |
| `--attack_samples` | 200 | Samples for attack evaluation |
| `--output_dir` | ./outputs | Output directory |
| `--seed` | 42 | Random seed |

## Methodology

### 1. Data Preprocessing
- Load AnnoCTR dataset (CTI reports with MITRE ATT&CK labels)
- Reduce to top 15 classes + OTHER
- Balance training data

### 2. Baseline Training
- Fine-tune SecBERT for sequence classification
- Use weighted cross-entropy for class imbalance

### 3. Adversarial Training (FGSM)
- Generate adversarial embeddings using Fast Gradient Sign Method
- Train on combined clean + adversarial loss (70/30 split)

### 4. Attack Evaluation
Multiple text-level attacks:
- **Synonym Attack**: Replace words with WordNet synonyms
- **Character Swap**: Swap adjacent characters
- **Homoglyph Attack**: Replace with Unicode lookalikes
- **Keyboard Typo**: Introduce proximity-based typos
- **Combined**: Apply multiple attacks

### 5. Metrics
- **Clean Accuracy**: Performance on original texts
- **Robust Accuracy**: Performance on adversarial texts
- **Attack Success Rate (ASR)**: % of correct predictions flipped by attack

## Generated Visualizations

The framework generates comprehensive visualizations:

1. **Training History** - Loss and metrics over epochs
2. **Label Distribution** - Class balance visualization
3. **Confusion Matrices** - Side-by-side baseline vs adversarial
4. **Robustness Comparison** - Clean/robust accuracy and ASR
5. **Attack Comparison** - Performance across different attack types
6. **Per-Class Robustness** - ASR breakdown by class
7. **Text Length Analysis** - Document length distributions

## Module Usage

### Using Individual Components

```python
# Data preprocessing
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(base_path="/path/to/AnnoCTR")
train_df, dev_df, test_df = preprocessor.process()

# Model
from src.models.classifier import CTIClassifier

classifier = CTIClassifier(model_name="jackaduma/SecBERT", num_labels=16)

# Training
from src.training.trainer import BaselineTrainer, AdversarialTrainer

trainer = BaselineTrainer(classifier, train_loader, dev_loader)
history = trainer.train()

# Attacks
from src.attacks.text_attacks import CombinedAttack

attack = CombinedAttack()
adversarial_text = attack("Original CTI report text...")

# Evaluation
from src.evaluation.evaluator import AdversarialEvaluator

evaluator = AdversarialEvaluator(classifier)
results = evaluator.evaluate_text_attacks(test_df)

# Visualization
from src.visualization.plots import Visualizer

viz = Visualizer(save_dir="./outputs/figures")
viz.plot_robustness_comparison(baseline_metrics, adversarial_metrics)
```

## Results

| Metric | Baseline | FGSM-Trained |
|--------|----------|--------------|
| Original Accuracy | 0.450 | 0.200 |
| Robust Accuracy | 0.440 | 0.210 |
| Attack Success Rate | 1.5% | 0.5% |
| **ASR Reduction** | - | **66.67%** |

## Authors

- [Dana Al Rijjal](https://github.com/daaalrijjal)
- [Jouri Al Daghma](https://github.com/Jourialdagh)

**Supervised by:** Dr. Naila Marir – Effat University

## License

This project is licensed under the MIT License.

## Citation

If you use this code, please cite:

```bibtex
@misc{alrijjal2024adversarial,
  title={Adversarial Robustness for CTI NLP Models},
  author={Al Rijjal, Dana and Al Daghma, Jouri},
  year={2024},
  publisher={GitHub},
  url={https://github.com/daaalrijjal/Adversarial-Robustness-CTI}
}
```

## Acknowledgments

- [AnnoCTR Dataset](https://github.com/boschresearch/anno-ctr-lrec-coling-2024) by Bosch Research
- [SecBERT](https://huggingface.co/jackaduma/SecBERT) pretrained model
