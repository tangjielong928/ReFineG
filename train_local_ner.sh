#!/bin/bash

config_path="./Local_NER/examples/CCKS2025/ccks2025_500.yaml"

python Local_NER/scripts/train.py -c ${config_path}
