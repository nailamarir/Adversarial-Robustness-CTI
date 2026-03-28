# A3L: Adaptive Adversarial Active Learning for CTI Classification

A multi-agent framework for adversarially robust cyber threat intelligence (CTI) classification using MITRE ATT&CK technique labels.

## Overview

Machine learning models deployed in CTI pipelines are vulnerable to adversarial manipulation. Standard adversarial training requires labeling large volumes of adversarial examples --- a costly process requiring expert security analysts. **A3L** addresses this by formulating adversarial defense as a *budget-constrained optimization problem*, selecting only the most informative adversarial samples for annotation.

The framework coordinates four specialized agents in a closed-loop adaptive defense cycle:

- **Detection Agent** --- identifies adversarial candidates via FGSM embedding attacks and multi-attack text perturbations
- **Selection Agent** --- ranks candidates by Adversarial Sample Utility (composite of loss, confidence flip, and margin)
- **Retraining Agent** --- performs interleaved adversarial training with epsilon curriculum
- **Audit Agent** --- monitors robustness metrics and adapts budget/threshold parameters

## Project Structure

```
.
├── configs/
│   └── config.py                 # Hyperparameter configuration
├── src/
│   ├── agents/                   # Four-agent framework
│   │   ├── detection_agent.py    # Adversarial pool generation (FGSM + text attacks)
│   │   ├── selection_agent.py    # Utility-based sample selection
│   │   ├── retraining_agent.py   # Interleaved adversarial training
│   │   ├── audit_agent.py        # Robustness monitoring and adaptation
│   │   └── framework.py          # Orchestrator (Algorithm 1)
│   ├── attacks/
│   │   ├── fgsm.py               # FGSM and PGD embedding-level attacks
│   │   └── text_attacks.py       # Text-level attacks (synonym, charswap, homoglyph, BERT-Attack, etc.)
│   ├── data/
│   │   ├── preprocessing.py      # AnnoCTR dataset loading and label reduction
│   │   ├── preprocessing_enhanced.py
│   │   └── dataset.py            # PyTorch DataLoader wrappers
│   ├── evaluation/
│   │   ├── evaluator.py          # Clean and adversarial evaluation
│   │   └── metrics.py            # ASR, robust accuracy, label efficiency
│   ├── models/
│   │   └── classifier.py         # CTIClassifier (DistilBERT, SecBERT, SecRoBERTa)
│   ├── training/
│   │   ├── trainer.py            # Baseline and adversarial trainers
│   │   └── losses.py             # Weighted CE, focal loss, TRADES
│   ├── utils/
│   │   └── helpers.py            # Seed, device, timing utilities
│   └── visualization/
│       └── plots.py              # Visualization methods
├── run_all_experiments.py         # Unified experiment runner (all models, seeds, ablations)
├── run_evaluation.py              # Legacy evaluation script
├── setup.py
└── requirements.txt
```

## Dataset

This project uses the [AnnoCTR](https://github.com/boschresearch/anno-ctr-lrec-coling-2024) corpus of labeled English-language cyber threat intelligence reports aligned with MITRE ATT&CK techniques (15 classes + OTHER).

```bash
git clone https://github.com/boschresearch/anno-ctr-lrec-coling-2024.git AnnoCTR_repo
mv AnnoCTR_repo/AnnoCTR ./AnnoCTR
rm -rf AnnoCTR_repo
```

## Models

Three transformer architectures evaluated:

| Model | HuggingFace ID | Parameters | Domain |
|-------|---------------|------------|--------|
| DistilBERT | `distilbert-base-uncased` | 66M | General |
| SecBERT | `jackaduma/SecBERT` | 110M | Cybersecurity |
| SecRoBERTa | `jackaduma/SecRoBERTa` | 125M | Cybersecurity |

## Installation

```bash
python -m venv venv
source venv/bin/activate

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Dependencies
pip install transformers pandas scikit-learn nltk matplotlib seaborn tqdm
```

## Usage

### Quick Test (single model, single seed)

```bash
python run_all_experiments.py \
    --data_path ./AnnoCTR \
    --output_dir ./outputs/quick_test \
    --model distilbert \
    --quick
```

### Full Experiment (all models, 5 seeds)

```bash
python run_all_experiments.py \
    --data_path ./AnnoCTR \
    --output_dir ./outputs/paper_final \
    --model all \
    --seeds 42 123 456 789 1024
```

### Single Model

```bash
python run_all_experiments.py \
    --data_path ./AnnoCTR \
    --output_dir ./outputs/secbert \
    --model secbert \
    --seeds 42
```

## Key Methods

### Defense Strategies Compared

1. **No Defense** --- clean training baseline
2. **Full Adversarial Training** --- FGSM on all training samples (upper bound)
3. **Random Selection** --- random AL baseline
4. **Pure Entropy** --- highest prediction entropy
5. **Pure Margin** --- smallest top-2 class margin
6. **Core-Set** --- greedy k-center diversity sampling
7. **Static-Budget AL** --- entropy selection without adaptive mechanisms
8. **Entropy + Core-Set** --- hybrid uncertainty-diversity
9. **A3L Composite** --- full framework with Adversarial Sample Utility scoring

### Six Training Enhancements

1. Interleaved adversarial training (no sequential phase forgetting)
2. FGSM on random clean batches (label-budget compliant)
3. Epsilon curriculum [0.02, 0.04, 0.06, 0.08, 0.10]
4. Progressive regularization (weak early, strong late)
5. Adversarial Sample Utility scoring (loss + confidence flip + margin)
6. Multi-attack pool generation (FGSM embedding + text-level attacks)

## Evaluation Metrics

- **Clean Accuracy**: accuracy on unperturbed test samples
- **Robust Accuracy**: accuracy under FGSM attack (epsilon=0.1)
- **Attack Success Rate (ASR)**: fraction of correct predictions flipped
- **Label Efficiency**: robust accuracy gain per labeled sample

## Citation

Paper under review. Citation details will be added upon publication.

## License

This project is for research purposes. The AnnoCTR dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Authors

- Naila Marir (Effat University)
- Dana Alrijal (Effat University)
- Jouri Aldaghma (Effat University)
