"""Operational detection metrics.

A respectable ROC-AUC can still produce an unusable alert stream, which is the
failure this project already hit. These are the measurements an analyst
actually experiences: how many false alerts arrive per hour, how much recall
survives a fixed alert budget, and how clean the top of the queue is.

Every metric here is prevalence-dependent, so any caller evaluating on a
subsample must pass the sampling rate rather than silently reporting a number
scaled by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

DEFAULT_TOP_K = (10, 50, 100)
FALSE_POSITIVE_BUDGET = "false_positive_budget"
BENIGN_QUANTILE = "benign_quantile"


class BudgetUnreachableError(ValueError):
    """Raised when no threshold can meet the configured alert budget.

    Refusing is deliberate. A bundle whose anomaly threshold is whatever a
    search loop happened to leave behind is worse than no bundle at all.
    """


def precision_recall_auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
    """Average precision, which is informative where ROC-AUC is not."""

    truth = np.asarray(truth)
    if not truth.any() or truth.all():
        return None
    return float(average_precision_score(truth, np.asarray(scores)))


def false_positives_per_hour(
    truth: np.ndarray,
    flagged: np.ndarray,
    benign_hours: float,
    sampling_rate: float = 1.0,
) -> float:
    """Benign rows flagged, per hour of benign traffic.

    `sampling_rate` scales the count back up to the population. Counting false
    positives in a sample while dividing by a full-capture denominator reports
    the rate low by exactly that factor, which would make a budget-driven
    threshold let through 1/rate times the intended volume.
    """

    if benign_hours <= 0:
        raise ValueError("benign_hours must be positive to report a rate")
    if not 0 < sampling_rate <= 1:
        raise ValueError("sampling_rate must be in (0, 1]")

    truth = np.asarray(truth)
    flagged = np.asarray(flagged, dtype=bool)
    false_positives = int(np.count_nonzero(flagged & (truth == 0)))
    return float(false_positives / sampling_rate / benign_hours)


def recall_at_false_positive_budget(
    truth: np.ndarray,
    scores: np.ndarray,
    budget_per_hour: float,
    benign_hours: float,
    sampling_rate: float = 1.0,
) -> tuple[float, float]:
    """The recall still available once the alert budget is enforced.

    Candidates are the observed benign scores plus one just above the highest
    of them. That last candidate flags no benign row at all, so a budget is
    always technically meetable and this function always returns a threshold.
    Without it a perfectly separating detector is rejected: if every benign row
    ties at one score, the highest real candidate still flags all of them, and
    a budget that a slightly higher threshold would meet with zero false
    positives looks unreachable.

    Whether the resulting recall is worth shipping is a policy question, and it
    belongs to select_anomaly_threshold rather than to this measurement.
    """

    truth = np.asarray(truth)
    scores = np.asarray(scores, dtype=float)
    positives = truth != 0
    benign_scores = scores[~positives]
    if not benign_scores.size:
        raise ValueError("Cannot budget false positives without benign rows")

    observed = np.unique(benign_scores)
    candidates = np.append(observed, np.nextafter(observed[-1], np.inf))
    for threshold in candidates:
        flagged = scores >= threshold
        rate = false_positives_per_hour(truth, flagged, benign_hours, sampling_rate)
        if rate <= budget_per_hour:
            recall = (
                float(np.count_nonzero(flagged & positives) / np.count_nonzero(positives))
                if positives.any()
                else 0.0
            )
            return recall, float(threshold)

    # The final candidate flags nothing, so this is unreachable in practice.
    raise BudgetUnreachableError(
        f"No threshold meets {budget_per_hour} false positives per hour, not "
        "even one above every benign score. This should be impossible; the "
        "inputs are probably not what this function expects."
    )


def top_k_precision(
    truth: np.ndarray,
    scores: np.ndarray,
    k: int,
    sampling_rate: float = 1.0,
) -> tuple[float | None, float]:
    """Precision over the K highest-scoring rows, the analyst triage view.

    Returns the precision and the effective population K, because the top K of
    a sample at rate r corresponds to roughly the top k/r of the population.
    """

    truth = np.asarray(truth)
    scores = np.asarray(scores, dtype=float)
    effective_k = float(k / sampling_rate)
    if k > truth.size:
        return None, effective_k

    top = np.argsort(scores)[::-1][:k]
    return float(np.count_nonzero(truth[top] != 0) / k), effective_k


def expected_calibration_error(
    truth: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Gap between stated confidence and observed accuracy."""

    truth = np.asarray(truth)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == truth).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if not in_bin.any():
            continue
        error += float(in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean()))
    return error


@dataclass(frozen=True)
class ThresholdPolicy:
    """Why a threshold is what it is, recorded alongside the threshold."""

    policy: str
    parameter: float
    threshold: float
    realized_false_positives_per_hour: float | None
    benign_hours: float
    sampling_rate: float

    def to_manifest(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "parameter": self.parameter,
            "threshold": self.threshold,
            "realized_false_positives_per_hour": self.realized_false_positives_per_hour,
            "benign_hours": self.benign_hours,
            "sampling_rate": self.sampling_rate,
        }


def select_anomaly_threshold(
    policy: str,
    *,
    truth: np.ndarray,
    scores: np.ndarray,
    benign_hours: float,
    budget_per_hour: float | None = None,
    quantile: float | None = None,
    sampling_rate: float = 1.0,
) -> ThresholdPolicy:
    """Choose the anomaly threshold from an operational constraint.

    The default policy is an alert budget rather than a ROC-derived cutoff,
    because the deployment constraint is how many alerts a person can read.
    """

    truth = np.asarray(truth)
    scores = np.asarray(scores, dtype=float)

    if policy == FALSE_POSITIVE_BUDGET:
        if budget_per_hour is None:
            raise ValueError("budget_per_hour is required for the budget policy")
        recall, threshold = recall_at_false_positive_budget(
            truth, scores, budget_per_hour, benign_hours, sampling_rate
        )
        # A budget that can only be met by flagging nothing is met, and
        # useless. Refusing here is deliberate: shipping a bundle that detects
        # nothing within its own alert budget is worse than shipping none.
        if recall <= 0.0 and np.any(truth != 0):
            raise BudgetUnreachableError(
                f"A budget of {budget_per_hour} false positives per hour can "
                "only be met by flagging nothing: no threshold catches a single "
                "attack row within it. Raise the budget or improve the benign "
                "separation before exporting a bundle."
            )
        realized = false_positives_per_hour(
            truth, scores >= threshold, benign_hours, sampling_rate
        )
        return ThresholdPolicy(
            policy=policy,
            parameter=float(budget_per_hour),
            threshold=float(threshold),
            realized_false_positives_per_hour=realized,
            benign_hours=float(benign_hours),
            sampling_rate=float(sampling_rate),
        )

    if policy == BENIGN_QUANTILE:
        if quantile is None:
            raise ValueError("quantile is required for the quantile policy")
        benign_scores = scores[truth == 0]
        if not benign_scores.size:
            raise ValueError("Cannot calibrate a benign quantile without benign rows")
        threshold = float(np.quantile(benign_scores, quantile))
        realized = false_positives_per_hour(
            truth, scores >= threshold, benign_hours, sampling_rate
        )
        return ThresholdPolicy(
            policy=policy,
            parameter=float(quantile),
            threshold=threshold,
            realized_false_positives_per_hour=realized,
            benign_hours=float(benign_hours),
            sampling_rate=float(sampling_rate),
        )

    raise ValueError(
        f"Unknown threshold policy {policy!r}; expected {FALSE_POSITIVE_BUDGET!r} "
        f"or {BENIGN_QUANTILE!r}"
    )


def operational_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    anomaly_scores: np.ndarray,
    classes: Sequence[str],
    *,
    benign_hours: float,
    benign_hours_source: str,
    anomaly_threshold: float,
    sampling_rate: float = 1.0,
    top_k_values: Sequence[int] = DEFAULT_TOP_K,
    min_family_support: int = 50,
) -> dict[str, Any]:
    """Assemble the operational block written into the bundle manifest."""

    targets = np.asarray(targets)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)
    benign_index = list(classes).index("Benign")
    binary_truth = (targets != benign_index).astype(int)
    flagged = anomaly_scores >= anomaly_threshold

    families: dict[str, Any] = {}
    for index, label in enumerate(classes):
        if label == "Benign":
            continue
        support = int(np.count_nonzero(targets == index))
        family_truth = (targets == index).astype(int)
        # A PR-AUC resting on a handful of positives is noise wearing a number.
        value = (
            precision_recall_auc(family_truth, probabilities[:, index])
            if support >= min_family_support
            else None
        )
        families[label] = {"pr_auc": value, "support": support}

    top_k: dict[str, Any] = {}
    for k in top_k_values:
        precision, effective_k = top_k_precision(
            binary_truth, anomaly_scores, k, sampling_rate
        )
        top_k[f"top_{k}"] = {"precision": precision, "effective_k": effective_k}

    return {
        "prevalence_basis": {
            "sampling": "uniform",
            "rate": float(sampling_rate),
            "rows_seen": int(targets.size),
        },
        "benign_hours": float(benign_hours),
        "benign_hours_source": benign_hours_source,
        "anomaly_pr_auc": precision_recall_auc(binary_truth, anomaly_scores),
        "false_positives_per_hour": false_positives_per_hour(
            binary_truth, flagged, benign_hours, sampling_rate
        ),
        "expected_calibration_error": expected_calibration_error(targets, probabilities),
        "top_k_precision": top_k,
        "per_family": families,
        "min_family_support": int(min_family_support),
    }


__all__ = [
    "BENIGN_QUANTILE",
    "BudgetUnreachableError",
    "DEFAULT_TOP_K",
    "FALSE_POSITIVE_BUDGET",
    "ThresholdPolicy",
    "expected_calibration_error",
    "false_positives_per_hour",
    "operational_metrics",
    "precision_recall_auc",
    "recall_at_false_positive_budget",
    "select_anomaly_threshold",
    "top_k_precision",
]
