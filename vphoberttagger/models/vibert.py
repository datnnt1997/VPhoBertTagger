from transformers import logging, BertForTokenClassification
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from .base import NerOutput

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.set_verbosity_error()


class viBertSoftmax(BertForTokenClassification):
    def __init__(self, config, **kwargs):
        super(viBertSoftmax, self).__init__(config=config, **kwargs)
        self.num_labels = config.num_labels

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        seq_output = self.bert(input_ids=input_ids,
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
            return NerOutput(loss=loss, tags=seq_tags)
        else:
            return NerOutput(tags=seq_tags)


class viBertCrf(BertForTokenClassification):
    def __init__(self, config):
        super(viBertCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.crf = CRF(config.num_labels, batch_first=True)
        self.init_weights()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        seq_outputs = self.bert(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   head_mask=None)[0]

        batch_size, max_len, feat_dim = seq_outputs.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_outputs.device).unsqueeze(1)
        seq_outputs = seq_outputs[range_vector, valid_ids]
        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)
        seq_tags = self.crf.decode(logits, mask=label_masks != 0)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_masks.type(torch.uint8))
            return NerOutput(loss=-1.0 * log_likelihood, tags=seq_tags)
        else:
            return NerOutput(tags=seq_tags)


class viBertLstmCrf(BertForTokenClassification):
    def __init__(self, config):
        super(viBertLstmCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)

    @staticmethod
    def sort_batch(src_tensor, lengths):
        """
        Sort a minibatch by the length of the sequences with the longest sequences first
        return the sorted batch targes and sequence lengths.
        This way the output can be used by pack_padded_sequences(...)
        """
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = src_tensor[perm_idx]
        _, reversed_idx = perm_idx.sort(0, descending=False)
        return seq_tensor, seq_lengths, reversed_idx

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        seq_outputs = self.bert(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                head_mask=None)[0]

        batch_size, max_len, feat_dim = seq_outputs.shape
        seq_lens = torch.sum(label_masks, dim=-1)
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_outputs.device).unsqueeze(1)
        seq_outputs = seq_outputs[range_vector, valid_ids]

        sorted_seq_outputs, sorted_seq_lens, reversed_idx = self.sort_batch(src_tensor=seq_outputs,
                                                                            lengths=seq_lens)
        packed_words = pack_padded_sequence(sorted_seq_outputs, sorted_seq_lens.cpu(), True)
        lstm_outs, _ = self.lstm(packed_words)
        lstm_outs, _ = pad_packed_sequence(lstm_outs, batch_first=True, total_length=max_len)
        seq_outputs = lstm_outs[reversed_idx]

        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)
        seq_tags = self.crf.decode(logits, mask=label_masks != 0)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_masks.type(torch.uint8))
            return NerOutput(loss=-1.0 * log_likelihood, tags=seq_tags)
        else:
            return NerOutput(tags=seq_tags)


# DEBUG
if __name__ == "__main__":
    from transformers import BertConfig

    model_name = 'vinai/phobert-base'
    config = BertConfig.from_pretrained(model_name, num_labels=7)
    model = viBertLstmCrf.from_pretrained(model_name, config=config, from_tf=False)

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
