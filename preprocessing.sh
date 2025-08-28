#!/bin/bash

raw_json_path="data/evaluation_test_set/Evaluation_text.json"
output_path="data/CCKS_NER_Aug/CCKS_test.conll"

python data_preprocess.py \
  --raw_json_path "$raw_json_path" \
  --output_path "$output_path" 