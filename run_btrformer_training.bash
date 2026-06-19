#!/bin/bash

# Exit immediately if a command fails
set -e

echo "======================================================="
echo "      Starting BTRFormer Training Pipeline             "
echo "======================================================="

# Step 1: Run the Training Command
echo "[INFO] Commencing model training. This may take a while..."
python main_ft.py \
    --cfg config.py \
    --opts DATA.DATASET imagenet \
    MODEL.TYPE swin \
    MODEL.NAME btrformer \
    TRAIN.LR_SCHEDULER.NAME cosine

echo "======================================================="
echo "      Training Complete! Saving checkpoints...         "
echo "======================================================="

# Step 2: Run the Evaluation Command
# We look for epoch 29 assuming a 30-epoch training cycle (0-29).
CHECKPOINT_PATH="output/btrformer/default/ckpt_epoch_29.pth"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "[INFO] Found completed checkpoint: $CHECKPOINT_PATH"
    echo "[INFO] Commencing Evaluation on test data..."
    
    python main_ft.py \
        --cfg config.py \
        --eval \
        --resume "$CHECKPOINT_PATH" \
        --opts DATA.DATASET imagenet MODEL.TYPE swin MODEL.NAME btrformer

    echo "======================================================="
    echo "      Evaluation Complete! Pipeline Finished.          "
    echo "======================================================="
else
    echo "[WARNING] Final checkpoint ($CHECKPOINT_PATH) not found."
    echo "[WARNING] Skipping automatic evaluation."
fi