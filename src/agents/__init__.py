"""
Multi-Agent Defense Framework
=============================
Agentic Active Learning-Enhanced Adaptive Defense Framework
for Cyber Threat Intelligence Classification.

Four specialized agents operate in a closed feedback loop:
1. Detection Agent - Perceives incoming inputs and flags adversarial anomalies
2. Selection Agent - Scores candidates by uncertainty and selects top-B samples
3. Retraining Agent - Performs incremental adversarial fine-tuning
4. Audit Agent - Monitors system behavior and generates explainable decision logs
"""

from .detection_agent import DetectionAgent
from .selection_agent import SelectionAgent
from .retraining_agent import RetrainingAgent
from .audit_agent import AuditAgent
from .framework import AgenticDefenseFramework

__all__ = [
    "DetectionAgent",
    "SelectionAgent",
    "RetrainingAgent",
    "AuditAgent",
    "AgenticDefenseFramework",
]
