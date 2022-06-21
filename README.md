# <div align="center">🍜VPhoBertTagger</div>

Token classification using Phobert Models for 🇻🇳Vietnamese

## <div align="center">🏞️Environments🏞️</div>
Get started in seconds with verified environments. Run script below for install all dependencies
```bash
bash ./install_dependencies.sh
```
## <div align="center">📚Dataset📚</div>
The input data's format of 🍜**VPhoBertTagger** follows [**VLSP-2016**](https://vlsp.org.vn/vlsp2016/eval/ner) format with four columns separated by a tab character, 
including of **word**, **pos**, **chunk**, and **named entity**. Each word which was segmented has been put on a separate line and there is 
an empty line after each sentence. For details, see sample data in **'datasets/samples'** directory. The table below describes an 
example Vietnamese sentence in dataset.


| Word         | POS | Chunk | NER   |
|--------------|-----|-------|-------|
| Dương	       |Np	  |B-NP	  |B-PER  |
| là	          |V	  |B-VP	  |O      |
| một	         |M	  |B-NP	  |O      |
| chủ       	  |N	  |B-NP	  |O      |
| cửa hàng   	 |N	  |B-NP	  |O      |
| lâu	         |A	  |B-AP	  |O      |
| năm	         |N	  |B-NP	  |O      |
| ở	           |E	  |B-PP	  |O      |
| Hà Nội 	     |Np	  |B-NP	  |B-LOC  |
| .	           |CH	  |O	  |O      |

The dataset must put on directory with structure as below.
```text
├── data_dir
|  └── train.txt
|  └── dev.txt
|  └── test.txt
```

## <div align="center">🎓Training🎓</div>
The commands below fine-tune **PhoBert** for Token-classification task. [Models](https://github.com/VinAIResearch/PhoBERT) download automatically from the latest
Hugging Face [release](https://huggingface.co/vinai)
```bash
python main.py train --task vlsp2016 --run_test --data_dir ./datasets/samples --model_name_or_path vinai/phobert-base --model_arch softmax --output_dir outputs --max_seq_length 256 --train_batch_size 32 --eval_batch_size 32 --learning_rate 5e-5 --epochs 3 --overwrite_data
```

or

```bash
bash ./train.sh
```

> Arguments:
> + ***type*** (`str`,`*required`): What is process type to be run. Must in [`train`, `test`, `predict`].
> + ***task*** (`str`, `*optional`): Training task selected in the list: [`vlsp2016`, `vlsp2018_l1`, `vlsp2018_l2`, `vlsp2018_join`]. Default: `vlsp2016`
> + ***data_dir*** (`Union[str, os.PathLike]`, `*required`): The input data dir. Should contain the .csv files (or other data files) for the task.
> + ***overwrite_data*** (`bool`, `*optional`) : Whether not to overwirte splitted dataset. Default=False
> + ***load_weights*** (`Union[str, os.PathLike]`, `*optional`): Path of pretrained file.
> + ***model_name_or_path*** (`str`, `*required`): Pre-trained model selected in the list: [`vinai/phobert-base`, `vinai/phobert-large`,...] Default=`vinai/phobert-base` 
> + ***model_arch*** (`str`, `*required`): Punctuation prediction model architecture selected in the list: [`softmax`, `crf`, `lstm_crf`].
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
> + ***run_test*** (`bool`, `*optional`): Whether not to run evaluate best model on test set after train. Default=False.
> + ***seed*** (`bool`, `*optional`): Random seed for initialization. Default=42.
> + ***num_workers*** (`int`, `*optional`): how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default=0.
> + ***save_step*** (`int`, `*optional`): The number of steps in the model will be saved. Default=10000.
> + ***gradient_accumulation_steps*** (`int`, `*optional`): Number of updates steps to accumulate before performing a backward/update pass. Default=1.

## <div align="center">🧠Inference🧠</div>
The command below load your fine-tuned model and inference in your text input.
```bash
python main.py predict --model_path outputs/best_model.pt
```

> Arguments:
> + ***type*** (`str`,`*required`): What is process type to be run. Must in [`train`, `test`, `predict`].
> + ***model_path*** (`Union[str, os.PathLike]`, `*optional`): Path of pretrained file.
> + ***no_cuda*** (`bool`, `*optional`): Whether not to use CUDA when available. Default=False.

## <div align="center">💡Acknowledgements💡</div>
Pretrained model [Phobert](https://github.com/VinAIResearch/PhoBERT) by [VinAI Research](https://github.com/VinAIResearch) and Pytorch implementation by [Hugging Face](https://huggingface.co/).

