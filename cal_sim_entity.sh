#!/bin/bash

eval_json="checkpoints/tjl928/xlm_roberta_large_best_0810/pred.txt"
eval_text="data/evaluation_test_set/Evaluation_text.json"
sample_json="data/sample_entity.json"
output="similarity_top3/output_top3/similarity_matrix_entity.json"
model_name="models/sentence-transformers/all-MiniLM-L6-v2"
batch_size=64
device="cuda"

python ./similarity_top3/util_calculate_similarity_entity.py \
  --eval_json "$eval_json" \
  --eval_text "$eval_text" \
  --sample_json "$sample_json" \
  --output "$output" \
  --model_name "$model_name" \
  --batch_size "$batch_size" \
  --device "$device"
   