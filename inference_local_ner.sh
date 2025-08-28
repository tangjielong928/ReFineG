#!/bin/bash

checkpoint_dir="./checkpoints/tjl928/xlm_roberta_large_best_0810"

python Local_NER/scripts/test.py -w ${checkpoint_dir}
