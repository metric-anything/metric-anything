"""
depth – Metric depth estimation wrappers
=========================================
Provides :class:`DAv2Estimator` (fast, Depth-Anything-V2) and
:class:`MetricAnythingEstimator` (accurate, MoGe student_pointmap).
"""

from .dav2 import DAv2Estimator
from .metric_anything import MetricAnythingEstimator

__all__ = ["DAv2Estimator", "MetricAnythingEstimator"]
