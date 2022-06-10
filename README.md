<h1 align="center">🍜VPhoBertNer</h1>

Named entity recognition using Phobert Models for 🇻🇳Vietnamese

## Train process
```bash
python trainer.py train --data_dir ./datasets/samples --model_name_or_path vinai/phobert-base --model_arch softmax --output_dir outputs --max_seq_length 256 --train_batch_size 32 --eval_batch_size 32 --learning_rate 5e-5 --epochs 3
```

or

```bash
bash ./train.sh
```

> Arguments:
> + ***type*** (`str`,`*required`): What is process type to be run. Must in [`train`, `test`].
> + ***task*** (`str`, `*optional`): Training task selected in the list: [`viner`]. Default: `viner`
> + ***data_dir*** (`Union[str, os.PathLike]`, `*required`): The input data dir. Should contain the .csv files (or other data files) for the task.
> + ***overwrite_data*** (`bool`, `*optional`) : Whether not to overwirte splitted dataset. Default=False
> + ***load_weights*** (`Union[str, os.PathLike]`, `*optional`): Path of pretrained file.
> + ***model_name_or_path*** (`str`, `*required`): Pre-trained model selected in the list: bert-base-uncased, bert-base-cased... Default=bert-base-multilingual-cased 
> + ***model_arch*** (`str`, `*required`): Punctuation prediction model architecture selected in the list: [`softmax`, `lstm_crf`].
> + ***output_dir*** (`Union[str, os.PathLike]`, `*required`): The output directory where the model predictions and checkpoints will be written.
> + ***max_seq_length*** (`int`, `*optional`): The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Default=190.
> + ***train_batch_size*** (`int`, `*optional`): Total batch size for training. Default=32.
> + ***eval_batch_size*** (`int`, `*optional`): Total batch size for eval. Default=32.
> + ***learning_rate*** (`float`, `*optional`): The initial learning rate for Adam. Default=5e-5.
> + ***epochs*** (`float`, `*optional`): Total number of training epochs to perform. Default=100.0.
> + ***weight_decay*** (`float`, `*optional`): Weight deay if we apply some. Default=0.01.
> + ***adam_epsilon*** (`float`, `*optional`): Epsilon for Adam optimizer. Default=5e-8.
> + ***max_grad_norm*** (`float`, `*optional`): Max gradient norm. Default=1.0.
> + ***early_stop*** (`float`, `*optional`): Number of early stop step. Default=10.0.
> + ***no_cuda*** (`bool`, `*optional`): Whether not to use CUDA when available. Default=False.
> + ***seed*** (`bool`, `*optional`): Random seed for initialization. Default=42.
> + ***num_workers*** (`int`, `*optional`): how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default=0.
> + ***save_step*** (`int`, `*optional`): The number of steps in the model will be saved. Default=10000.
> + ***gradient_accumulation_steps*** (`int`, `*optional`): Number of updates steps to accumulate before performing a backward/update pass. Default=1.

## Predict process
```bash
python predictor.py predict --model_path outputs/best_model.pt
```

> Arguments:
> + ***type*** (`str`,`*required`): What is process type to be run. Must in [`predict`].
> + ***load_weights*** (`Union[str, os.PathLike]`, `*optional`): Path of pretrained file.
> + ***no_cuda*** (`bool`, `*optional`): Whether not to use CUDA when available. Default=False.

