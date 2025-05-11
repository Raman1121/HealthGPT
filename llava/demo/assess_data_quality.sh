#!/bin/bash

set -e
set -o pipefail

METADATA_CSV="/pvc/SynthCheX/sana_epoch50/generations_with_metadata.csv"
IMAGE_DIR="/pvc/SynthCheX/sana_epoch50/"
OUTPUT_DIR="/pvc/SynthCheX/"

IMAGE_COL="synthetic_filename"
CAPTION_COL="annotated_prompt"
LABELS_COL="chexpert_labels"

NUM_SHARDS=5
SHARD=1

# Path to the HLORA and Fusion Layer files
HLORA_PATH="/pvc/HealthGPT_model_weights/com_hlora_weights.bin"
FUSION_LAYER_PATH="/pvc/HealthGPT_model_weights/fusion_layer_weights.bin"

python3 com_infer.py \
    --model_name_or_path="microsoft/Phi-3-mini-4k-instruct" \
    --dtype "FP16" --hlora_r "64" --hlora_alpha "128" --hlora_nums "4" \
    --vq_idx_nums "8192" --instruct_template "phi3_instruct" \
    --hlora_path $HLORA_PATH --fusion_layer_path $FUSION_LAYER_PATH \
    --metadata_csv $METADATA_CSV \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --num_shards $NUM_SHARDS \
    --shard $SHARD \
    --image_col $IMAGE_COL \
    --caption_col $CAPTION_COL \
    --labels_col $LABELS_COL \