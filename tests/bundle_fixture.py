"""Shared helper for building a small, valid v2 model bundle in tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from detector_runtime import HybridDetectorNetwork
from explain import EXPLANATION_METHOD, EXPLANATION_VERSION
from feature_contract import (
    BUNDLE_FORMAT_VERSION,
    FEATURE_CONTRACT_VERSION,
    TRANSFORM,
    WINDOW_SECONDS,
)

FLOW_FEATURES = ["Flow Duration", "Tot Fwd Pkts"]
WINDOW_FEATURES = ["win1s_dst_flow_count", "win60s_dst_unique_src"]


def write_bundle(
    directory: Path,
    *,
    with_window_block: bool = False,
    anomaly_threshold: float = 0.1,
    disabled_reason: str | None = None,
    decoder_gain: float = 0.0,
    classifier_gain: float = 0.0,
) -> dict:
    """Write a zero-weight bundle whose behaviour is fully predictable."""

    window = list(WINDOW_FEATURES) if with_window_block else []
    features = list(FLOW_FEATURES) + window

    model = HybridDetectorNetwork(
        input_dim=len(features),
        num_classes=2,
        hidden_dims=(4,),
        latent_dim=2,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    if decoder_gain:
        # An all-zero decoder reconstructs zero, which makes every per-feature
        # attribution equal to the raw input and any ranking assertion vacuous.
        with torch.no_grad():
            model.encoder[1].weight.fill_(1.0)
            model.decoder[2].weight.copy_(
                torch.eye(len(features), 4) * float(decoder_gain)
            )
    if classifier_gain:
        # With an all-zero classifier every softmax is uniform, so no row ever
        # clears a family threshold and the named-family path is unreachable.
        with torch.no_grad():
            model.encoder[1].weight.fill_(1.0)
            model.encoder[0].weight[0, features.index("Tot Fwd Pkts")] = 1.0
            model.encoder[0].bias.copy_(torch.tensor([0.5, -0.25, 0.75, -1.0]))
            model.encoder[4].weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
            model.classifier.weight.copy_(
                torch.tensor([[0.0, 0.0], [float(classifier_gain), 0.0]])
            )
    torch.save(model.state_dict(), directory / "model.pt")

    manifest = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "feature_contract": {
            "version": FEATURE_CONTRACT_VERSION,
            "features": features,
            "flow_features": list(FLOW_FEATURES),
            "window_features": window,
            "window_seconds": list(WINDOW_SECONDS) if with_window_block else [],
            "window_features_disabled_reason": disabled_reason,
            "transform": TRANSFORM,
            "mean": [0.0] * len(features),
            "scale": [1.0] * len(features),
            "missing_value_policy": "reject",
        },
        "taxonomy": {
            "classes": ["Benign", "DDoS"],
            "reportable": {"Benign": True, "DDoS": True},
            "unknown_label": "Unknown anomaly",
        },
        "model": {"hidden_dims": [4], "latent_dim": 2, "dropout": 0.0},
        "thresholds": {
            "anomaly": anomaly_threshold,
            "class_confidence": {"Benign": 0.0, "DDoS": 0.9},
        },
        "explanation": {
            "version": EXPLANATION_VERSION,
            "methods": EXPLANATION_METHOD,
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest
