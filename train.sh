#!/usr/bin/env bash

pip install -r requirements.txt
export PYTHONPATH=.
python3 trainer.py train train --data_dir datasets/samples --model_name_or_path vinai/phobert-base --model_arch softmax --output_dir outputs --max_seq_length 256 --train_batch_size 1 --eval_batch_size 1 --learning_rate 5e-5 --epochs 3