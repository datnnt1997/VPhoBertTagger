from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset

from vphoberttagger.processor import NerFeatures
from vphoberttagger.constant import PROCESSOR_MAPPING

import os
import torch


class NerDataset(Dataset):
    def __init__(self, features: List[NerFeatures], device: str = 'cpu'):
        self.examples = features
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: val.to(self.device) for key, val in self.examples[index].__dict__.items()}


def build_dataset(data_dir: Union[str, os.PathLike],
                  tokenizer,
                  label2id: List[str],
                  header: List[str],
                  dtype: str = 'train',
                  max_seq_len:int = 256,
                  device:str = 'cpu',
                  use_crf: bool = False,
                  overwrite_data: bool = False) -> NerDataset:
    if header == 'jsonl':
        process_key = tokenizer.name_or_path+'/jsonl'
        dfile_path = Path(data_dir + f'/{dtype}.jsonl')
    else:
        process_key = tokenizer.name_or_path
        dfile_path = Path(data_dir+f'/{dtype}.txt')
    cached_path = dfile_path.with_suffix('.cached')
    if not os.path.exists(cached_path) or overwrite_data:
        features = PROCESSOR_MAPPING[process_key](dfile_path, tokenizer, label2id, header, max_seq_len, use_crf=use_crf)
        torch.save(features, cached_path)
    else:
        features = torch.load(cached_path)
    return NerDataset(features=features, device=device)


# DEBUG
if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from vphoberttagger.constant import LABEL2ID_VLSP2018
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    ner_dataset = build_dataset('../datasets/vlsp2018',
                                tokenizer,
                                dtype='train',
                                max_seq_len=128,
                                device='cuda',
                                use_crf=True,
                                header=['token', 'tmp1', 'ner', 'tmp2'],
                                overwrite_data=True,
                                label2id=LABEL2ID_VLSP2018)
    ner_iterator = DataLoader(ner_dataset, batch_size=12)
    for batch in ner_iterator:
        print(batch)
        break
