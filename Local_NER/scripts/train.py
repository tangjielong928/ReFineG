# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import sys
import warnings

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from adaseq.commands.train import train_model_from_args, train_model  # noqa # isort:skip

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument(
        '-c', '--config_path', type=str, required=True, help='configuration YAML file'
    )
    parser.add_argument(
        '-w',
        '--work_dir',
        type=str,
        default=None,
        help='directory to save experiment logs and checkpoints',
    )
    parser.add_argument('-n', '--run_name', type=str, default=None, help='trial name')
    parser.add_argument(
        '-f', '--force', default=None, help='overwrite the output directory if it exists.'
    )
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, help='model checkpoint to load')
    parser.add_argument('--seed', type=int, default=None, help='random seed for everything')
    parser.add_argument('-d', '--device', type=str, default='gpu', help='device name')
    parser.add_argument('--use_fp16', action='store_true', help='whether to use mixed precision')
    parser.add_argument('--local_rank', type=str, default='0')

    args = parser.parse_args()
    train_model_from_args(args)
