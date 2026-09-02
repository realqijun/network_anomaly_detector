#!/bin/bash

MODEL_BUNDLE_SOURCE="${1:-model_bundle}"
DEPLOY_BASE_DIR="deploy"
DEPLOY_BUNDLE_DIR="${DEPLOY_BASE_DIR}/model_bundle"
TEMP_BUNDLE_DIR="${DEPLOY_BASE_DIR}/.model_bundle.tmp"

BUNDLE_FILES=(
    "manifest.json"
    "model.pt"
)

if [ ! -d "$MODEL_BUNDLE_SOURCE" ]; then
    echo "ERROR: Model bundle directory '$MODEL_BUNDLE_SOURCE' not found."
    echo "Usage: $0 [path-to-model-bundle]"
    exit 1
fi

rm -rf "$TEMP_BUNDLE_DIR"
mkdir -p "$TEMP_BUNDLE_DIR"
COPY_SUCCESS=0
for FILE in "${BUNDLE_FILES[@]}"; do
    SOURCE_PATH="${MODEL_BUNDLE_SOURCE}/${FILE}"
    DEST_PATH="${TEMP_BUNDLE_DIR}/${FILE}"

    if [ -f "$SOURCE_PATH" ]; then
        cp -v "$SOURCE_PATH" "$DEST_PATH"
        if [ $? -ne 0 ]; then
            echo "WARNING: Failed to copy '$FILE'."
            COPY_SUCCESS=1
        fi
    else
        echo "WARNING: Source bundle file '$SOURCE_PATH' not found. Skipping."
        COPY_SUCCESS=1
    fi
done

if [ $COPY_SUCCESS -eq 0 ]; then
    rm -rf "$DEPLOY_BUNDLE_DIR"
    mv "$TEMP_BUNDLE_DIR" "$DEPLOY_BUNDLE_DIR"
    echo "SUCCESS: Model bundle copied to ${DEPLOY_BUNDLE_DIR}."
    echo "Next step: docker build -f deploy/Dockerfile -t network-anomaly-detector ."
    exit 0
else
    rm -rf "$TEMP_BUNDLE_DIR"
    echo "WARNING: Some bundle files could not be copied. Please check the logs above."
    exit 1
fi
