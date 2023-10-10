from typing import List

import torch
from torch import nn


class ConvDecoder(nn.Module):
    def __init__(self, conf):

        super().__init__()

        self.labels = conf.labels
        self.blank_id = len(self.labels)
        self.layers = nn.Sequential(
            nn.Conv1d(conf.feat_in, len(self.labels) + 1, kernel_size=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        logits = self.layers(x)  # B x D x T
        logprobs = nn.functional.log_softmax(logits.transpose(1, 2), dim=-1)

        return logprobs

    def ids_to_tokens(self, ids) -> List[str]:
        return [self.labels[int(token_id.item())] for token_id in ids]

    def unique_consecutive(self, hypothesis):
        prev = self.blank_id

        token_ids = []

        for token_id in hypothesis:
            if token_id not in (prev, self.blank_id):
                token_ids.append(token_id)
            prev = token_id

        return token_ids

    def decode_hypothesis(self, token_ids, unique_consecutive: bool) -> str:
        if unique_consecutive:
            token_ids = self.unique_consecutive(token_ids)
        return "".join(self.ids_to_tokens(token_ids))

    def decode(
        self, token_ids, token_ids_length, unique_consecutive=False
    ) -> List[str]:
        return [
            self.decode_hypothesis(ids[:length], unique_consecutive)
            for ids, length in zip(token_ids, token_ids_length)
        ]
