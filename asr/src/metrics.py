from typing import List

import editdistance
import torch
from torchmetrics import Metric


class WER(Metric):
    def __init__(self):
        super().__init__()

        self.add_state(
            "word_errors",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
            persistent=False,
        )
        self.add_state(
            "words", default=torch.tensor(0), dist_reduce_fx="sum", persistent=False
        )

    def update(self, references: List[str], hypotheses: List[str]):
        word_errors = 0.0
        words = 0.0
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            dist = editdistance.eval(ref_tokens, hyp_tokens)
            word_errors += dist
            words += len(ref_tokens)
        self.word_errors = torch.tensor(
            word_errors, device=self.word_errors.device, dtype=self.word_errors.dtype
        )
        self.words = torch.tensor(
            words, device=self.words.device, dtype=self.words.dtype
        )

    def compute(self):
        word_errors = self.word_errors.detach().float()
        words = self.words.detach().float()
        return self.word_errors / self.words, word_errors, words
