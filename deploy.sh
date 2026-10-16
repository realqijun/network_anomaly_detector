#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/deploy"
IMAGE_NAME="${IMAGE_NAME:-network-anomaly-detector:local}"

required_artifacts=(
    "anomaly_detector_model.pth"
    "min_max_scaler.pkl"
    "optimal_threshold.npy"
)

for artifact in "${required_artifacts[@]}"; do
    artifact_path="${DEPLOY_DIR}/working/${artifact}"
    if [[ ! -f "${artifact_path}" ]]; then
        echo "ERROR: Required deployment artifact is missing: ${artifact_path}" >&2
        exit 1
    fi
done

docker build --tag "${IMAGE_NAME}" "${DEPLOY_DIR}"

echo "Built verified deployment image: ${IMAGE_NAME}"
