#!/bin/bash
#
# Stage a trained model bundle into deploy/model_bundle for the Docker build.
#
# Paths are resolved from this script's own location, never from $PWD. An
# earlier version used a bare "deploy" relative path, so running it from any
# other directory silently deleted whatever lived at $PWD/deploy/model_bundle
# and still reported success.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_BASE_DIR="${SCRIPT_DIR}/deploy"
DEPLOY_BUNDLE_DIR="${DEPLOY_BASE_DIR}/model_bundle"

BUNDLE_FILES=(
    "manifest.json"
    "model.pt"
)

usage() {
    echo "Usage: $0 [path-to-model-bundle]"
    echo "Copies manifest.json and model.pt into ${DEPLOY_BUNDLE_DIR}."
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

MODEL_BUNDLE_SOURCE="${1:-model_bundle}"

if [ ! -d "$DEPLOY_BASE_DIR" ]; then
    echo "ERROR: '${DEPLOY_BASE_DIR}' not found. This script must live in the repository root."
    exit 1
fi

if [ ! -d "$MODEL_BUNDLE_SOURCE" ]; then
    echo "ERROR: Model bundle directory '$MODEL_BUNDLE_SOURCE' not found."
    usage
    exit 1
fi

# Refuse a source inside the deploy directory. Staging would otherwise delete
# the very files being copied, then report them as "not found".
SOURCE_REAL="$(cd -- "$MODEL_BUNDLE_SOURCE" && pwd -P)"
DEPLOY_REAL="$(cd -- "$DEPLOY_BASE_DIR" && pwd -P)"
case "$SOURCE_REAL" in
    "$DEPLOY_REAL"|"$DEPLOY_REAL"/*)
        echo "ERROR: Source '$MODEL_BUNDLE_SOURCE' is inside '${DEPLOY_BASE_DIR}'."
        echo "Staging would delete it. Copy the bundle from outside deploy/ instead."
        exit 1
        ;;
esac

for FILE in "${BUNDLE_FILES[@]}"; do
    if [ ! -f "${SOURCE_REAL}/${FILE}" ]; then
        echo "ERROR: Source bundle file '${MODEL_BUNDLE_SOURCE}/${FILE}' not found."
        echo "A deployable bundle needs both ${BUNDLE_FILES[*]}."
        exit 1
    fi
done

# Stage into a fresh temporary directory so a half-finished copy can never be
# mistaken for a bundle, and so an existing bundle survives a failed run.
TEMP_BUNDLE_DIR="$(mktemp -d "${DEPLOY_BASE_DIR}/.model_bundle.XXXXXXXX")"
cleanup() {
    rm -rf "$TEMP_BUNDLE_DIR"
}
trap cleanup EXIT

for FILE in "${BUNDLE_FILES[@]}"; do
    cp -v "${SOURCE_REAL}/${FILE}" "${TEMP_BUNDLE_DIR}/${FILE}"
done

# Swap rather than delete-then-move: the previous bundle is only discarded
# once the replacement is in place.
PREVIOUS_BUNDLE_DIR=""
if [ -d "$DEPLOY_BUNDLE_DIR" ]; then
    PREVIOUS_BUNDLE_DIR="$(mktemp -d "${DEPLOY_BASE_DIR}/.model_bundle.old.XXXXXXXX")"
    rmdir "$PREVIOUS_BUNDLE_DIR"
    mv "$DEPLOY_BUNDLE_DIR" "$PREVIOUS_BUNDLE_DIR"
fi

if ! mv "$TEMP_BUNDLE_DIR" "$DEPLOY_BUNDLE_DIR"; then
    if [ -n "$PREVIOUS_BUNDLE_DIR" ]; then
        mv "$PREVIOUS_BUNDLE_DIR" "$DEPLOY_BUNDLE_DIR"
        echo "ERROR: Could not install the new bundle. The previous one was restored."
    else
        echo "ERROR: Could not install the new bundle."
    fi
    exit 1
fi

if [ -n "$PREVIOUS_BUNDLE_DIR" ]; then
    rm -rf "$PREVIOUS_BUNDLE_DIR"
fi

echo "SUCCESS: Model bundle copied to ${DEPLOY_BUNDLE_DIR}."
echo "Next step: docker build -f deploy/Dockerfile -t network-anomaly-detector ${SCRIPT_DIR}"
