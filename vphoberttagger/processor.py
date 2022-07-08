from typing import Union, List
from tqdm import tqdm

import os
import torch
import jsonlines
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


def convert_syllable_examples_features(data_path: Union[str, os.PathLike],
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
            token = row.token.replace(' ', '_').strip().split("_")
            tokens.extend(token)
            if row.ner.strip() == "O":
                tag_id = label2id.index(row.ner.strip())
                tag_ids.extend([tag_id] * len(token))
            elif row.ner.strip().startswith("I-"):
                tag_id = label2id.index(row.ner.strip())
                tag_ids.extend([tag_id] * len(token))
            elif row.ner.strip().startswith("B-"):
                _, tag = row.ner.strip().split("-")
                for idx, t in enumerate(token):
                    if idx == 0 :
                        tag_ids.append(label2id.index(f"B-{tag}"))
                    else:
                        tag_ids.append(label2id.index(f"I-{tag}"))
            else:
                raise f"{row} id not match case!"
            if not row_idx == len(data) - 1:
                continue
        assert len(tokens) == len(tag_ids), f"{tokens} and {tag_ids} not match!!"
        seq_len = len(tag_ids)
        encoding = tokenizer(tokens,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len,
                             is_split_into_words=True)

        valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
        label_marks = np.zeros(len(encoding.input_ids), dtype=int)
        valid_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
        i = 1
        word_ids = encoding.word_ids()
        for idx in range(1, len(word_ids)):
            if word_ids[idx] is not None and word_ids[idx] != word_ids[idx - 1]:
                if use_crf:
                    valid_ids[i-1] = idx
                else:
                    valid_ids[idx] = 1
                valid_labels[idx] = tag_ids[word_ids[idx]]
                i += 1
        if max_seq_len >= seq_len:
            label_padding_size = (max_seq_len - seq_len)
            label_marks[:seq_len] = [1] * seq_len
            tag_ids.extend([0] * label_padding_size)
        else:
            tag_ids = tag_ids[:max_seq_len]
            label_marks[:-2] = [1] * (max_seq_len - 2)
            tag_ids[-2:] = [0] * 2
        if use_crf and label_marks[0] == 0:
            raise f"{' '.join(tokens)} - {tag_ids} have mark == 0 at index 0"
        items = {key: val for key, val in encoding.items()}
        items['labels'] = tag_ids if use_crf else valid_labels
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_marks if use_crf else valid_ids

        features.append(NerFeatures(**items))

        for k, v in items.items():
            assert len(v) == max_seq_len, f"Expected length of {k} is {max_seq_len} but got {len(v)}"

        tokens = []
        tag_ids = []
    return features


def convert_word_segment_examples_features(data_path: Union[str, os.PathLike],
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
        if max_seq_len >= seq_len:
            label_padding_size = (max_seq_len - seq_len)
            label_marks[:seq_len] = [1] * seq_len
            tag_ids.extend([0] * label_padding_size)
        else:
            tag_ids = tag_ids[:max_seq_len]
            label_marks[:-2] = [1] * (max_seq_len - 2)
            tag_ids[-2:] = [0] * 2
        if use_crf and label_marks[0] == 0:
            raise f"{sentence} - {tag_ids} have mark == 0 at index 0!"

        items = {key: val for key, val in encoding.items()}
        items['labels'] = tag_ids if use_crf else valid_labels
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_marks if use_crf else valid_ids
        features.append(NerFeatures(**items))

        for k, v in items.items():
            assert len(v) == max_seq_len, f"Expected length of {k} is {max_seq_len} but got {len(v)}"

        tokens = []
        tag_ids = []
    return features


def convert_word_segment_examples_features_from_jsonl(data_path: Union[str, os.PathLike],
                                                      tokenizer,
                                                      label2id,
                                                      header_names: List[str],
                                                      max_seq_len: int = 256,
                                                      use_crf: bool = False) -> List[NerFeatures]:
    features = []
    with jsonlines.open(file=data_path, mode='r') as f:
        for line in f.iter():
            sentence = line['data']
            tokens = []
            sorted_labels = sorted(line['label'], key=lambda tup: tup[0])
            tag_ids = []
            last_index = 0
            # ii = 0
            for l in sorted_labels:
                tag = l[-1].strip()
                # tag_ids.extend([label2id.index('O')] * len(sentence[last_index: l[0]].strip().split()))

                for i, w in enumerate(sentence[last_index: l[0]].strip().split()) :
                    tokens.append(w)
                    tag_ids.append(label2id.index('O'))
                    # print(f"{tokens[ii]}\t{w}\tO")
                    # ii+=1

                for i, w in enumerate(sentence[l[0]: l[1]].strip().split()):
                    tokens.append(w)
                    if i == 0:
                        # print(f"{tokens[ii]}\t{w}\t{f'B-{tag}'}")
                        tag_ids.append(label2id.index(f'B-{tag}'))
                    else:
                        # print(f"{tokens[ii]}\t{w}\t{f'I-{tag}'}")
                        tag_ids.append(label2id.index(f'I-{tag}'))
                    # ii += 1
                last_index = l[1]
            for i, w in enumerate(sentence[last_index:].strip().split()):
                tokens.append(w)
                tag_ids.append(label2id.index('O'))
            # tag_ids.extend([label2id.index('O')] * len(sentence[last_index:].strip().split()))
            assert len(tokens) == len(tag_ids), f'[ERROR] {line["id"]} is not matching ' \
                                                f'number of tokens-{len(tokens)} and tags-{len(tag_ids)}: {line}'
            seq_len = len(tag_ids)
            encoding = tokenizer(sentence,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=max_seq_len)
            subwords = tokenizer.tokenize(sentence)
            valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
            label_marks = np.zeros(len(encoding.input_ids), dtype=int)
            valid_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
            i = 1
            for idx, subword in enumerate(subwords[:max_seq_len - 2]):
                if idx != 0 and subwords[idx - 1].endswith("@@"):
                    continue
                if use_crf:
                    valid_ids[i - 1] = idx + 1
                else:
                    valid_ids[idx + 1] = 1
                valid_labels[idx + 1] = tag_ids[i - 1]
                i += 1

            if max_seq_len > seq_len:
                label_padding_size = (max_seq_len - seq_len)
                label_marks[:seq_len] = [1] * seq_len
                tag_ids.extend([0] * label_padding_size)
            else:
                tag_ids = tag_ids[:max_seq_len]
                label_marks[:-2] = [1] * (max_seq_len - 2)
                tag_ids[-2:] = [0] * 2
            if use_crf and label_marks[0] == 0:
                raise f"{sentence} - {tag_ids} have mark == 0 at index 0!"

            items = {key: val for key, val in encoding.items()}
            items['labels'] = tag_ids if use_crf else valid_labels
            items['valid_ids'] = valid_ids
            items['label_masks'] = label_marks if use_crf else valid_ids
            features.append(NerFeatures(**items))

            for k, v in items.items():
                assert len(v) == max_seq_len, f"{line['id']} Expected length of {k} is {max_seq_len} but got {len(v)}"
        return features


# DEBUG
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from vphoberttagger.constant import LABEL_MAPPING
    label_info = LABEL_MAPPING['bds2022']
    convert_word_segment_examples_features_from_jsonl(data_path='./datasets/bds2022/test.jsonl',
                                                      tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base'),
                                                      label2id=label_info["label2id"],
                                                      header_names=label_info['header'],
                                                      max_seq_len=128,
                                                      use_crf=True)



