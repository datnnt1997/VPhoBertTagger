from model import PhoBertSoftmax
from arguments import get_predict_argument
from constant import LABEL2ID, ID2LABEL
from helper import normalize_text
from vncorenlp import VnCoreNLP

from typing import Union
from transformers import AutoConfig, AutoTokenizer

import os
import torch
import itertools
import numpy as np


class PhobertNER(object):
    def __init__(self, model_path: Union[str or os.PathLike],  no_cuda=False):
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
        print("[VPhoBertNer] VnCoreNLP loading ...")
        self.rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
        print("[VPhoBertNer] Model loading ...")
        self.model, self.tokenizer,  self.max_seq_len = self.load_model(model_path, device=self.device)
        print("[VPhoBertNer] All ready!")

    @staticmethod
    def load_model(model_path: Union[str or os.PathLike],  device='cpu'):
        if device == 'cpu':
            checkpoint_data = torch.load(model_path, map_location='cpu')
        else:
            checkpoint_data = torch.load(model_path)
        args = checkpoint_data["args"]
        max_seq_len = args.max_seq_length

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(LABEL2ID))

        model = PhoBertSoftmax(config=config)
        model.load_state_dict(checkpoint_data['model'])
        model.to(device)
        model.eval()

        return model, tokenizer, max_seq_len

    def preprocess(self, in_raw: str):
        norm_text = normalize_text(in_raw)
        sents = []
        sentences = self.rdrsegmenter.tokenize(norm_text)
        for sentence in sentences:
            sents.append(" ".join(sentence))
        return sents

    def convert_tensor(self, sent):
        encoding = self.tokenizer(sent,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_seq_len)
        subwords = self.tokenizer.tokenize(sent)
        valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
        i = 1
        for idx, subword in enumerate(subwords):
            if idx != 0 and subwords[idx - 1].endswith("@@"):
                continue
            valid_ids[idx + 1] = 1
            i += 1
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
                tag = ID2LABEL[l]
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


if __name__ == "__main__":
    args = get_predict_argument()
    predictor = PhobertNER(args.model_path, no_cuda=args.no_cuda)
    while True:
        in_raw = input('Enter text:')
        print(predictor(in_raw))

