from typing import Optional, List, Tuple, Any
from collections import OrderedDict

from transformers import logging, RobertaForTokenClassification
from torchcrf import CRF

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.set_verbosity_error()


class PhoBertNerOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = torch.FloatTensor([0.0])
    tags: Optional[List[int]] = []

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())


class PhoBertSoftmax(RobertaForTokenClassification):
    def __init__(self, config, **kwargs):
        super(PhoBertSoftmax, self).__init__(config=config, **kwargs)
        self.num_labels = config.num_labels

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        seq_output = self.roberta(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  head_mask=None)[0]
        seq_output = self.dropout(seq_output)
        logits = self.classifier(seq_output)
        probs = F.log_softmax(logits, dim=2)
        label_masks = label_masks.view(-1) != 0
        seq_tags = torch.masked_select(torch.argmax(probs, dim=2).view(-1), label_masks).tolist()
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
            return PhoBertNerOutput(loss=loss, tags=seq_tags)
        else:
            return PhoBertNerOutput(tags=seq_tags)


class PhoBertLstmCrf(RobertaForTokenClassification):
    def __init__(self, config):
        super(PhoBertLstmCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        seq_outputs = self.roberta(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   head_mask=None)[0]

        seq_outputs, _ = self.lstm(seq_outputs)

        batch_size, max_len, feat_dim = seq_outputs.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_outputs.device).unsqueeze(1)
        valid_seq_outputs = seq_outputs[range_vector, valid_ids]

        valid_seq_outputs = self.dropout(valid_seq_outputs)
        logits = self.classifier(valid_seq_outputs)

        seq_tags = self.crf.decode(logits, mask=label_masks != 0)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_masks.type(torch.uint8))
            return PhoBertNerOutput(loss=-1.0 * log_likelihood, tags=seq_tags)
        else:
            return PhoBertNerOutput(tags=seq_tags)


# DEBUG
if __name__ == "__main__":
    from transformers import RobertaConfig

    model_name = 'vinai/phobert-base'
    config = RobertaConfig.from_pretrained(model_name, num_labels=7)
    model = PhoBertLstmCrf.from_pretrained(model_name, config=config, from_tf=False)

    input_ids = torch.randint(0, 2999, [2, 20], dtype=torch.long)
    mask = torch.ones([2, 20], dtype=torch.long)
    labels = torch.randint(1, 6, [2, 20], dtype=torch.long)
    new_labels = torch.zeros([2, 20], dtype=torch.long)
    valid_ids = torch.ones([2, 20], dtype=torch.long)
    label_mask = torch.ones([2, 20], dtype=torch.long)
    valid_ids[:, 0] = 0
    valid_ids[:, 13] = 0
    labels[:, 0] = 0
    label_mask[:, -2:] = 0
    for i in range(len(labels)):
        idx = 0
        for j in range(len(labels[i])):
            if valid_ids[i][j] == 1:
                new_labels[i][idx] = labels[i][j]
                idx += 1
    output = model.forward(input_ids,
                           labels=new_labels,
                           attention_mask=mask,
                           valid_ids=valid_ids, label_masks=label_mask)
    print(new_labels)
    print(label_mask)
    print(valid_ids)

    print(output)
