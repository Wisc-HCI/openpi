"""Numerical summaries and explicit gates for the requested sanity tests."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats


@dataclasses.dataclass(frozen=True)
class PairedTestResult:
    count: int
    left_mean: float
    right_mean: float
    median_difference: float
    statistic: float
    pvalue: float
    passed: bool


def paired_lower_test(left: np.ndarray, right: np.ndarray, *, alpha: float = 0.05) -> PairedTestResult:
    """One-sided paired Wilcoxon test for the preregistered ``left < right`` direction."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired samples must be one-dimensional arrays with equal shape")
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 2:
        raise ValueError("At least two finite paired samples are required")
    difference = right - left
    try:
        test = stats.wilcoxon(left, right, alternative="less")
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    except ValueError:
        statistic, pvalue = 0.0, 1.0
    median_difference = float(np.median(difference))
    return PairedTestResult(
        count=len(left),
        left_mean=float(np.mean(left)),
        right_mean=float(np.mean(right)),
        median_difference=median_difference,
        statistic=statistic,
        pvalue=pvalue,
        passed=bool(pvalue < alpha and median_difference > 0.0),
    )
