"""
Utility Functions Module
Helper functions for the CTI adversarial robustness project
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as a readable string"""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{precision}f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def save_results(
    results: Dict[str, Any],
    filename: str,
    output_dir: str = "./outputs/logs"
) -> str:
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # Add timestamp
    results["timestamp"] = datetime.now().isoformat()

    filepath = os.path.join(output_dir, filename)

    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted section header"""
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Metrics Comparison"
) -> None:
    """Print metrics in a table format"""
    print_section(title)

    # Get all metric names
    all_metrics = set()
    for model_metrics in metrics.values():
        all_metrics.update(model_metrics.keys())

    # Header
    model_names = list(metrics.keys())
    header = f"{'Metric':<30}" + "".join(f"{name:>15}" for name in model_names)
    print(header)
    print("-" * len(header))

    # Rows
    for metric in sorted(all_metrics):
        row = f"{metric:<30}"
        for model in model_names:
            value = metrics[model].get(metric, "N/A")
            if isinstance(value, float):
                row += f"{value:>15.4f}"
            else:
                row += f"{str(value):>15}"
        print(row)


def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    """Get GPU memory usage information"""
    if not torch.cuda.is_available():
        return None

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")


class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        if self.name:
            print(f"{self.name}: {duration:.2f} seconds")
        else:
            print(f"Duration: {duration:.2f} seconds")

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
