#!/bin/bash

APP_SOURCE_DIR="."
SOURCE_MODELS_DIR="working"
DEPLOY_BASE_DIR="deploy"
DEPLOY_MODELS_DIR="${DEPLOY_BASE_DIR}/working"

MODEL_FILES=(
    "anomaly_detector_model.pth"
    "min_max_scaler.pkl"
    "optimal_threshold.npy"
)
APP_CODE_FILES=(
    "models.py"
)

# 1. Check if the source directory for models exists
if [ ! -d "$SOURCE_MODELS_DIR" ]; then
    echo "ERROR: Source directory '$SOURCE_MODELS_DIR/' not found."
    exit 1
fi

DATA_COPY_SUCCESS=0
for FILE in "${MODEL_FILES[@]}"; do
    SOURCE_PATH="${SOURCE_MODELS_DIR}/${FILE}"
    DEST_PATH="${DEPLOY_MODELS_DIR}/${FILE}"

    if [ -f "$SOURCE_PATH" ]; then
        cp -v "$SOURCE_PATH" "$DEST_PATH" # -v for verbose output
        if [ $? -ne 0 ]; then # Check if copy command was successful
            echo "WARNING: Failed to copy '$FILE'."
            DATA_COPY_SUCCESS=1
        fi
    else
        echo "WARNING: Source data file '$SOURCE_PATH' not found. Skipping."
        DATA_COPY_SUCCESS=1
    fi
done

CODE_COPY_SUCCESS=0
for FILE in "${APP_CODE_FILES[@]}"; do
    SOURCE_PATH="${APP_SOURCE_DIR}/${FILE}"
    DEST_PATH="${DEPLOY_BASE_DIR}/${FILE}"

    if [ -f "$SOURCE_PATH" ]; then
        cp -v "$SOURCE_PATH" "$DEST_PATH"
        if [ $? -ne 0 ]; then
            echo "WARNING: Failed to copy '$FILE'."
            CODE_COPY_SUCCESS=1
        fi
    else
        echo "WARNING: Source code file '$SOURCE_PATH' not found. Skipping."
        CODE_COPY_SUCCESS=1
    fi
done

if [ $DATA_COPY_SUCCESS -eq 0 ] && [ $CODE_COPY_SUCCESS -eq 0 ]; then
    echo "SUCCESS: All specified files were updated."
    exit 0
else
    echo "WARNING: Some files could not be copied. Please check the logs above."
    exit 1
fi