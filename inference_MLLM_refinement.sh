#!/bin/bash

IMG_PATH="./data/evaluation_test_set/Evaluation_image/"
INPUT_PATH="./data/evaluation_test_set/Evaluation_text.json"
PRED_PATH="./checkpoints/tjl928/xlm_roberta_large_best_0810/pred.txt"
ICL_DEMO_PATH="similarity_top3/output_top3/weighted_similarity_top3.json"
ICL_ANNOTATION_PATH="./data/sample_entity.json" 
ICL_IMAGE_PATH="./data/sample_image/"
OUTPUT_PATH="./output"
DATASET_NAME="CCKS_2025"
MODEL_NAME="qwen2.5-vl-72b-instruct"
NAME="3-shot_ICL"

python LLM_baseline/main.py \
  --img_path "$IMG_PATH" \
  --input_path "$INPUT_PATH" \
  --pred_path "$PRED_PATH" \
  --icl_demo_path "$ICL_DEMO_PATH" \
  --icl_annotation_path "$ICL_ANNOTATION_PATH" \
  --icl_image_path "$ICL_IMAGE_PATH" \
  --output_path "$OUTPUT_PATH" \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --name "$NAME" \
  --icl_flag
