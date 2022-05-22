from transformers import RobertaConfig, PhobertTokenizer
from model import PhoBertLstmCrf
from constant import LABEL2ID, ID2LABEL
from processor import normalize_text
from vncorenlp import VnCoreNLP

import os
import torch
import itertools
import numpy as np


class PhobertNER(object):
    def __init__(self, model_path: str = None, max_seq_length: int = 256, no_cuda=False):
        self.max_seq_len = max_seq_length
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
        self.rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
        self.model, self.tokenizer, =self.load_model(model_path, device=self.device)


    @staticmethod
    def load_model(model_path=None, model_clss='vinai/phobert-base', device='cpu'):
        tokenizer = PhobertTokenizer.from_pretrained(model_clss, use_fast=False)
        config = RobertaConfig.from_pretrained(model_clss, num_labels=len(LABEL2ID))
        model = PhoBertLstmCrf(config=config)
        if model_path is not None:
            if device == 'cpu':
                checkpoint_data = torch.load(model_path, map_location='cpu')
            else:
                checkpoint_data = torch.load(model_path)
            model.load_state_dict(checkpoint_data['model'])
        model.to(device)
        model.eval()
        return model, tokenizer

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
        valid_id = np.zeros(len(encoding["input_ids"]), dtype=int)
        i = 0
        subwords = self.tokenizer.tokenize(sent)
        for idx, sword in enumerate(subwords):
            if not sword.endswith('@@'):
                valid_id[idx+1] = 1
                i += 1
            elif idx == 0 or not subwords[idx-1].endswith('@@'):
                valid_id[idx + 1] = 1
                i += 1
            else:
                continue
        label_masks = [1] * i
        label_masks.extend([0] * (self.max_seq_len - len(label_masks)))
        encoding.pop('offset_mapping', None)
        item = {key: torch.as_tensor([val]).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['valid_ids'] = torch.as_tensor([valid_id]).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor([label_masks]).to(self.device, dtype=torch.long)
        return item

    def __call__(self, in_raw: str):
        sents = self.preprocess(in_raw)
        entites = []
        for sent in sents:
            item = self.convert_tensor(sent)
            with torch.no_grad():
                tags = self.model(**item)
            entity = None
            for w, l in list(zip(sent.split(), list(itertools.chain(*tags)))):
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
    predictor = PhobertNER()
    while True:
        in_raw = input('Enter text:')
        print(predictor(in_raw))

