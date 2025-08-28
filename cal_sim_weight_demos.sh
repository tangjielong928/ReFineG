#!/bin/bash

text_sim="similarity_top3/output_top3/similarity_matrix_text.json"
image_sim="similarity_top3/output_top3/similarity_matrix_image.json"
entity_sim="similarity_top3/output_top3/similarity_matrix_entity.json"
eval_text_json="data/evaluation_test_set/Evaluation_text.json"
sample_text_json="data/sample_text.json"
output="similarity_top3/output_top3/weighted_similarity_top3.json"
text_weight=0.4
image_weight=0.2
entity_weight=0.6
topk=3

python ./similarity_top3/util_calculate_similarity_weighted.py \
  --text_sim "$text_sim" \
  --image_sim "$image_sim" \
  --entity_sim "$entity_sim" \
  --eval_text_json "$eval_text_json" \
  --sample_text_json "$sample_text_json" \
  --output "$output" \
  --text_weight "$text_weight" \
  --image_weight "$image_weight" \
  --entity_weight "$entity_weight" \
  --topk "$topk"
   