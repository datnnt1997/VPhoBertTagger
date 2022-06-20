from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

import os
import torch
import pandas as pd
import numpy as np


class NerFeatures(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, valid_ids, labels, label_masks):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.token_type_ids = torch.as_tensor(token_type_ids, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long)
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long)


def convert_examples_features(data_path: Union[str, os.PathLike],
                              tokenizer,
                              label2id,
                              header_names: List[str],
                              max_seq_len: int = 256,
                              use_crf: bool = False) -> List[NerFeatures]:
    features = []
    tokens = []
    tag_ids = []
    data = pd.read_csv(data_path,
                       delimiter='\t',
                       encoding='utf-8',
                       skip_blank_lines=False,
                       names=header_names)
    data.fillna(method="ffill")
    for row_idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Load dataset {data_path}..."):
        if row.notna().token:
            tokens.append(row.token.strip().replace(' ', '_'))
            tag_ids.append(label2id.index(row.ner.strip()))
            if not row_idx == len(data) - 1:
                continue
        seq_len = len(tokens)
        sentence = ' '.join(tokens)
        encoding = tokenizer(sentence,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len)
        subwords = tokenizer.tokenize(sentence)
        valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
        label_marks = np.zeros(len(encoding.input_ids), dtype=int)
        valid_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
        i = 1
        for idx, subword in enumerate(subwords[:max_seq_len-2]):
            if idx != 0 and subwords[idx-1].endswith("@@"):
                continue
            if use_crf:
                valid_ids[i-1] = idx + 1
            else:
                valid_ids[idx+1] = 1
            valid_labels[idx+1] = tag_ids[i-1]
            i += 1
        if max_seq_len > seq_len + 2:
            label_padding_size = (max_seq_len - seq_len)
            label_marks[:seq_len] = [1] * seq_len
            tag_ids.extend([0] * label_padding_size)
        else:
            label_marks[:-2] = [1] * (max_seq_len - 2)
            tag_ids[-2:] = [0] * 2
        items = {key: val for key, val in encoding.items()}
        items['labels'] = tag_ids if use_crf else valid_labels
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_marks if use_crf else valid_ids
        features.append(NerFeatures(**items))
        tokens = []
        tag_ids = []
    return features


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
    dfile_path = Path(data_dir+f'/{dtype}.txt')
    cached_path = dfile_path.with_suffix('.cached')
    if not os.path.exists(cached_path) or overwrite_data:
        features = convert_examples_features(dfile_path, tokenizer, label2id, header, max_seq_len, use_crf=use_crf)
        torch.save(features, cached_path)
    else:
        features = torch.load(cached_path)
    return NerDataset(features=features, device=device)


# DEBUG
if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    ner_dataset = build_dataset('../datasets/samples',
                                tokenizer,
                                dtype='train',
                                max_seq_len=256,
                                device='cuda')
    ner_iterator = DataLoader(ner_dataset, batch_size=12)
    for batch in ner_iterator:
        print(batch)
        break
