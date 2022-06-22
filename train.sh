#!/usr/bin/env bash

export PYTHONPATH=.
python3 main.py train --task vlsp2016 --run_test --data_dir ./datasets/vlsp2016 --model_name_or_path vinai/phobert-base --model_arch softmax --output_dir outputs --max_seq_length 256 --train_batch_size 32 --eval_batch_size 32 --learning_rate 3e-5 --epochs 20 --early_stop 2 --overwrite_data