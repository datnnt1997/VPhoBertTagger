from argparse import ArgumentParser


def get_predict_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What processs to be run')
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def get_test_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What processs to be run')
    parser.add_argument("--data_dir", default='datasets/samples', type=str,
                        help="The input data dir. Should contain the .txt files (or other data files) for the task.")
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwirte splitted dataset")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--num_worker', type=int, default=2,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in "
                             "the main process.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def get_train_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What process to be run')
    parser.add_argument("--task", default='vlsp2016', type=str, choices=['vlsp2016', 'vlsp2018_l1', 'vlsp2018_l2',
                                                                         'vlsp2018_join', 'bds2022', 'covid19'],
                        help="Training task selected in the list: [`vlsp2016`, `vlsp2018_l1`, `vlsp2018_l2`, "
                             "`vlsp2018_join`].")
    parser.add_argument("--data_dir", default='datasets/samples', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwirte splitted dataset")
    parser.add_argument("--load_weights", default=None, type=str,
                        help='Path of pretrained file.')
    parser.add_argument("--model_name_or_path", default='vinai/phobert-base', type=str,
                        help="Pre-trained model selected in the list: vinai/phobert-base, vinai/phobert-large...")
    parser.add_argument("--model_arch", default='softmax', type=str, choices=['softmax', 'crf', 'lstm_crf'],
                        help="Punctuation prediction model architecture selected in the list: softmax, crf, lstm_crf")
    parser.add_argument("--output_dir", default='outputs/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--early_stop", default=10.0, type=float,
                        help="")
    parser.add_argument("--run_test", action='store_true', default=False,
                        help="Whether not to run test after train")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded "
                             "in the main process.")
    parser.add_argument('--save_step', type=int, default=20000,
                        help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser.parse_args()
