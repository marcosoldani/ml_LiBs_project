"""Numerical drift detection between a reference and a current distribution.

This module deliberately avoids heavy dependencies (no Alibi Detect, no
Evidently) so the project keeps a tiny footprint while still covering the
Lecture 06 syllabus. Two complementary tests are provided:

* **Kolmogorov–Smirnov (KS) test** — non-parametric two-sample test on the
  *empirical CDF*. It is sensitive to any change in shape and produces a
  p-value, so the threshold is interpretable.
* **Population Stability Index (PSI)** — bin-based divergence used heavily in
  industry for model monitoring. PSI < 0.1 ≈ no significant change,
  0.1 ≤ PSI < 0.25 ≈ minor drift, PSI ≥ 0.25 ≈ major drift.

Combining them avoids the false negatives of any single test: KS catches
shape changes, PSI catches mass-shifts between bins.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DEFAULT_PSI_BINS = 10
DEFAULT_KS_ALPHA = 0.05
DEFAULT_PSI_THRESHOLD = 0.25


@dataclass
class FeatureDrift:
    """Per-feature drift summary returned by :func:`detect_dataset_drift`."""

    feature: str
    n_reference: int
    n_current: int
    ks_statistic: float
    ks_p_value: float
    ks_drift: bool
    psi: float
    psi_drift: bool

    @property
    def drift(self) -> bool:
        """Combined verdict: drift detected if either test fires."""
        return bool(self.ks_drift or self.psi_drift)

    def to_dict(self) -> dict:
        body = asdict(self)
        body["drift"] = self.drift
        return body


@dataclass
class DriftReport:
    """Aggregate drift report across many features."""

    features: list[FeatureDrift] = field(default_factory=list)
    ks_alpha: float = DEFAULT_KS_ALPHA
    psi_threshold: float = DEFAULT_PSI_THRESHOLD

    @property
    def n_drifted(self) -> int:
        return sum(1 for f in self.features if f.drift)

    @property
    def share_drifted(self) -> float:
        return self.n_drifted / max(len(self.features), 1)

    @property
    def drift_detected(self) -> bool:
        """Conventionally we flag *dataset* drift when ≥ 25 % of features drift."""
        return self.share_drifted >= 0.25

    def to_dict(self) -> dict:
        return {
            "ks_alpha": self.ks_alpha,
            "psi_threshold": self.psi_threshold,
            "n_features": len(self.features),
            "n_drifted": self.n_drifted,
            "share_drifted": self.share_drifted,
            "drift_detected": self.drift_detected,
            "features": [f.to_dict() for f in self.features],
        }


def ks_drift(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = DEFAULT_KS_ALPHA,
) -> tuple[float, float, bool]:
    """Kolmogorov–Smirnov two-sample test.

    Returns ``(statistic, p_value, drifted)`` where ``drifted`` is ``True``
    when ``p_value < alpha`` (i.e. distributions differ significantly).
    """
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size == 0 or cur.size == 0:
        return float("nan"), float("nan"), False
    result = ks_2samp(ref, cur)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    return statistic, p_value, p_value < alpha


def psi_drift(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = DEFAULT_PSI_BINS,
    threshold: float = DEFAULT_PSI_THRESHOLD,
    epsilon: float = 1e-6,
) -> tuple[float, bool]:
    """Population Stability Index between two 1-D distributions.

    Bins are derived from the reference quantiles so that, *under no drift*,
    each bin holds ≈ ``1/bins`` of the mass. The PSI sums ``(p_i - q_i) *
    log(p_i / q_i)`` over the bins, with floors at ``epsilon`` to avoid log(0).
    """
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size == 0 or cur.size == 0:
        return float("nan"), False

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 3:
        edges = np.linspace(float(ref.min()), float(ref.max()) + epsilon, bins + 1)

    edges[0] -= epsilon
    edges[-1] += epsilon

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    p = ref_hist.astype(float) / max(ref_hist.sum(), 1)
    q = cur_hist.astype(float) / max(cur_hist.sum(), 1)
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    psi = float(np.sum((p - q) * np.log(p / q)))
    return psi, psi >= threshold


def detect_dataset_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Iterable[str] | None = None,
    ks_alpha: float = DEFAULT_KS_ALPHA,
    psi_bins: int = DEFAULT_PSI_BINS,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> DriftReport:
    """Run KS and PSI on every requested numerical feature.

    Parameters
    ----------
    reference, current
        DataFrames sharing at least the requested feature columns.
    features
        Subset of columns to analyse. Defaults to the numerical intersection of
        the two frames.
    """
    if features is None:
        numeric_ref = set(reference.select_dtypes(include="number").columns)
        numeric_cur = set(current.select_dtypes(include="number").columns)
        features = sorted(numeric_ref & numeric_cur)

    report = DriftReport(ks_alpha=ks_alpha, psi_threshold=psi_threshold)
    for feature in features:
        if feature not in reference or feature not in current:
            continue
        ref = reference[feature].to_numpy()
        cur = current[feature].to_numpy()
        ks_stat, ks_p, ks_flag = ks_drift(ref, cur, alpha=ks_alpha)
        psi, psi_flag = psi_drift(
            ref, cur, bins=psi_bins, threshold=psi_threshold
        )
        report.features.append(
            FeatureDrift(
                feature=feature,
                n_reference=int(np.isfinite(ref).sum()),
                n_current=int(np.isfinite(cur).sum()),
                ks_statistic=ks_stat,
                ks_p_value=ks_p,
                ks_drift=bool(ks_flag),
                psi=psi,
                psi_drift=bool(psi_flag),
            )
        )
    return report


def _clean(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]
