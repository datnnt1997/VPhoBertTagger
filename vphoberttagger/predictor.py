from vphoberttagger.arguments import get_predict_argument
from vphoberttagger.constant import LABEL_MAPPING, MODEL_MAPPING
from vphoberttagger.helper import normalize_text
from vncorenlp import VnCoreNLP

from typing import Union
from transformers import AutoConfig, AutoTokenizer

import os
import torch
import itertools
import numpy as np


class ViTagger(object):
    def __init__(self, model_path: Union[str or os.PathLike],  no_cuda=False):
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
        print("[ViTagger] VnCoreNLP loading ...")
        self.rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
        print("[ViTagger] Model loading ...")
        self.model, self.tokenizer,  self.max_seq_len, self.label2id, self.use_crf = self.load_model(model_path, device=self.device)
        self.id2label = {idx: label for idx, label in enumerate(self.label2id)}
        print("[ViTagger] All ready!")

    @staticmethod
    def load_model(model_path: Union[str or os.PathLike],  device='cpu'):
        if device == 'cpu':
            checkpoint_data = torch.load(model_path, map_location='cpu')
        else:
            checkpoint_data = torch.load(model_path)
        args = checkpoint_data["args"]
        max_seq_len = args.max_seq_length
        use_crf = True if 'crf' in args.model_arch else False
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(args.label2id))
        model_clss = MODEL_MAPPING[args.model_name_or_path][args.model_arch]
        model = model_clss(config=config)
        model.load_state_dict(checkpoint_data['model'])
        model.to(device)
        model.eval()

        return model, tokenizer, max_seq_len, args.label2id, use_crf

    def preprocess(self, in_raw: str):
        norm_text = normalize_text(in_raw)
        sents = []
        sentences = self.rdrsegmenter.tokenize(norm_text)
        for sentence in sentences:
            sents.append(sentence)
        return sents

    def convert_tensor(self, tokens):
        seq_len = len(tokens)
        encoding = self.tokenizer(tokens,
                                  padding='max_length',
                                  truncation=True,
                                  is_split_into_words=True,
                                  max_length=self.max_seq_len)
        if 'vinai/phobert' in self.tokenizer.name_or_path:
            subwords = self.tokenizer.tokenize(tokens, is_split_into_words=True)
            valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
            label_marks = np.zeros(len(encoding.input_ids), dtype=int)
            i = 1
            for idx, subword in enumerate(subwords[:self.max_seq_len - 2]):
                if idx != 0 and subwords[idx - 1].endswith("@@"):
                    continue
                if self.use_crf:
                    valid_ids[i - 1] = idx + 1
                else:
                    valid_ids[idx + 1] = 1
                i += 1
        else:
            valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
            label_marks = np.zeros(len(encoding.input_ids), dtype=int)
            i = 1
            word_ids = encoding.word_ids()
            for idx in range(1, len(word_ids)):
                if word_ids[idx] is not None and word_ids[idx] != word_ids[idx - 1]:
                    if self.use_crf:
                        valid_ids[i - 1] = idx
                    else:
                        valid_ids[idx] = 1
                    i += 1
        if self.max_seq_len >= seq_len + 2:
            label_marks[:seq_len] = [1] * seq_len
        else:
            label_marks[:-2] = [1] * (self.max_seq_len - 2)
        if self.use_crf and label_marks[0] == 0:
            raise f"{tokens} have mark == 0 at index 0!"
        item = {key: torch.as_tensor([val]).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['valid_ids'] = torch.as_tensor([valid_ids]).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor([valid_ids]).to(self.device, dtype=torch.long)
        return item

    def __call__(self, in_raw: str):
        sents = self.preprocess(in_raw)
        entites = []
        for sent in sents:
            item = self.convert_tensor(sent)
            with torch.no_grad():
                outputs = self.model(**item)
            entity = None
            if isinstance(outputs.tags, tuple):
                tags = list(itertools.chain(*outputs.tags))
            else:
                tags = outputs.tags
            for w, l in list(zip(sent.split(), tags)):
                tag = self.id2label[l]
                if not tag == 'O':
                    prefix, tag = tag.split('-')
                    if entity is None:
                        entity = (w, tag)
                    else:
                        if entity[-1] == tag:
                            if prefix == 'I':
                                entity = (entity[0] + f' {w}', tag)
                            else:
                                entites.append(entity)
                                entity = (w, tag)
                        else:
                            entites.append(entity)
                            entity = (w, tag)
                elif entity is not None:
                    entites.append(entity)
                    entity = None
                else:
                    entity = None
        return entites


def tagging():
    args = get_predict_argument()
    predictor = ViTagger(args.model_path, no_cuda=args.no_cuda)
    while True:
        in_raw = input('Enter text:')
        print(predictor(in_raw))


if __name__ == "__main__":
    tagging()

