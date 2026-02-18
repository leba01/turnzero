"""Evaluation harness: metrics, plots, and stratified reporting."""

from turnzero.eval.metrics import compute_metrics
from turnzero.eval.plots import reliability_diagram, topk_comparison_bar, stratified_table

__all__ = [
    "compute_metrics",
    "reliability_diagram",
    "topk_comparison_bar",
    "stratified_table",
]
