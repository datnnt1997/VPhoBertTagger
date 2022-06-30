from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split


def run():
    parser = ArgumentParser()
    parser.add_argument("--data_path", default='datasets/bds2022/data.jsonl', type=str,
                        help="The input data path. File must have formatted in jsonl.")
    parser.add_argument("--train_name", default='train.jsonl', type=str,
                        help="The train data file name")
    parser.add_argument("--test_name", default='test.jsonl', type=str,
                        help="The train data file name")
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--is_shuffle',  action='store_true', default=False,
                        help="Whether not to use shuffle before split dataset")

    args = parser.parse_args()

    lines = open(args.data_path, 'r', encoding='utf-8').readlines()

    train_lines, test_lines = train_test_split(lines,
                                               test_size=args.test_ratio,
                                               shuffle=args.is_shuffle,
                                               random_state=42)
    data_dir = Path(args.data_path).parent
    with open(data_dir/args.train_name, 'w', encoding='utf-8') as fw:
        fw.writelines(train_lines)
    with open(data_dir/args.test_name, 'w', encoding='utf-8') as fw:
        fw.writelines(test_lines)


if __name__ == "__main__":
    run()