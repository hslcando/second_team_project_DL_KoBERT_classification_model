import torch
from torch import nn
import numpy as np


class BERTSentenceTransform:

    def __init__(self, tokenizer, max_seq_length, pad=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad

    def __call__(self, sent):
        sent_tokens = self._tokenizer.tokenize(sent)
        if len(sent_tokens) > self._max_seq_length - 2:
            sent_tokens = sent_tokens[0 : (self._max_seq_length - 2)]

        tokens = []
        tokens.append("[CLS]")
        tokens.extend(sent_tokens)
        tokens.append("[SEP]")

        segment_ids = [0] * len(tokens)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        valid_length = len(input_ids)

        if self._pad:
            padding_length = self._max_seq_length - valid_length
            input_ids.extend([1] * padding_length)
            segment_ids.extend([0] * padding_length)

        return (
            np.array(input_ids, dtype="int32"),
            np.array(valid_length, dtype="int32"),
            np.array(segment_ids, dtype="int32"),
        )


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=4, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device),
            return_dict=False,
        )
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
