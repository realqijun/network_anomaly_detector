"""Feature attribution for detector signals.

A raw anomaly score is hard to trust and hard to debug. These explanations say
which contract features drove a score and how far each sat from the benign
baseline, so a false positive can be argued with rather than only observed.

Two bases, because the runtime produces two kinds of non-benign row and one
method does not honestly cover both:

- a named attack family comes from a class logit, so attribution is gradient
  times input against that logit;
- an unknown anomaly comes from reconstruction error and has no logit to
  differentiate, so attribution is the per-feature decomposition of the score
  itself, which is exactly faithful because the score is their mean.

These support analyst interpretation. They are not proof on their own.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

EXPLANATION_VERSION = 1

EXPLANATION_BASIS_CLASS = "gradient_x_input"
EXPLANATION_BASIS_ANOMALY = "reconstruction_error_decomposition"

EXPLANATION_METHOD = {
    EXPLANATION_BASIS_CLASS: (
        "Gradient of the predicted class logit with respect to the standardized "
        "feature, multiplied by that feature's value."
    ),
    EXPLANATION_BASIS_ANOMALY: (
        "Per-feature squared reconstruction error. The reported anomaly score is "
        "the mean of these terms."
    ),
}

DEFAULT_TOP_K = 5


def _rank(
    contributions: np.ndarray,
    observed: np.ndarray,
    directions: np.ndarray,
    feature_names: Sequence[str],
    basis: str,
    top_k: int,
) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(contributions))[::-1][:top_k]
    return [
        {
            "feature": feature_names[index],
            "contribution": float(contributions[index]),
            "observed_z": float(observed[index]),
            "direction": int(directions[index]),
            "basis": basis,
        }
        for index in order
    ]


def explain_rows(
    model: torch.nn.Module,
    prepared: np.ndarray,
    feature_names: Sequence[str],
    *,
    class_indices: np.ndarray,
    use_reconstruction: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    device: torch.device | None = None,
) -> list[list[dict[str, Any]]]:
    """Explain each prepared row against the basis its prediction came from.

    This runs its own forward pass under enable_grad over freshly built
    tensors. Detector.detect scores under torch.inference_mode, and a tensor
    produced there cannot have requires_grad set afterwards, so explanations
    cannot piggyback on the scoring loop.
    """

    prepared = np.ascontiguousarray(prepared, dtype=np.float32)
    if prepared.shape[0] == 0:
        return []

    class_indices = np.asarray(class_indices, dtype=np.int64)
    use_reconstruction = np.asarray(use_reconstruction, dtype=bool)
    device = device or next(model.parameters()).device
    explanations: list[list[dict[str, Any]]] = [[] for _ in range(prepared.shape[0])]

    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            features = torch.tensor(prepared, device=device, requires_grad=True)
            logits, reconstruction = model(features)

            reconstruction_rows = np.flatnonzero(use_reconstruction)
            if reconstruction_rows.size:
                error = (features - reconstruction).detach().cpu().numpy()
                for row in reconstruction_rows:
                    residual = error[row]
                    explanations[row] = _rank(
                        residual**2,
                        prepared[row],
                        np.sign(residual).astype(int),
                        feature_names,
                        EXPLANATION_BASIS_ANOMALY,
                        top_k,
                    )

            class_rows = np.flatnonzero(~use_reconstruction)
            if class_rows.size:
                selected = logits[
                    torch.arange(len(prepared), device=device),
                    torch.tensor(class_indices, device=device),
                ]
                gradients = torch.autograd.grad(selected.sum(), features)[0]
                attribution = (gradients * features).detach().cpu().numpy()
                for row in class_rows:
                    contributions = attribution[row]
                    explanations[row] = _rank(
                        contributions,
                        prepared[row],
                        np.sign(contributions).astype(int),
                        feature_names,
                        EXPLANATION_BASIS_CLASS,
                        top_k,
                    )
    finally:
        model.train(was_training)

    return explanations


__all__ = [
    "DEFAULT_TOP_K",
    "EXPLANATION_BASIS_ANOMALY",
    "EXPLANATION_BASIS_CLASS",
    "EXPLANATION_METHOD",
    "EXPLANATION_VERSION",
    "explain_rows",
]
