#!/bin/bash

eval_json="data/evaluation_test_set/Evaluation_image"
sample_json="data/sample_image"
output="similarity_top3/output_top3/similarity_matrix_image.json"
model_name="models/sentence-transformers/clip-ViT-L-14"
batch_size=32
device="cuda"

python ./similarity_top3/util_calculate_similarity_image.py \
  --eval_img_dir "$eval_json" \
  --sample_img_dir "$sample_json" \
  --output "$output" \
  --model_name "$model_name" \
  --batch_size "$batch_size" \
  --device "$device"
   