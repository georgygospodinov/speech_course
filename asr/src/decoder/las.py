import copy

import sentencepiece
import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=conf.tokenizer)

        self.embedding = nn.Embedding(self.tokenizer.vocab_size(), conf.embedding_dim)
        self.pos_embedding = PositionalEncoding(
            conf.embedding_dim, dropout=conf.dropout
        )

        layer = nn.TransformerDecoderLayer(
            conf.embedding_dim,
            conf.n_heads,
            conf.d_model,
            conf.dropout,
            batch_first=True,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(conf.n_layers)]
        )

        self.out = nn.Linear(conf.embedding_dim, self.tokenizer.vocab_size())

    def forward(
        self,
        encoded: torch.Tensor,
        encoded_pad_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        target_pad_mask: torch.Tensor,
    ):
        emb = self.pos_embedding(self.embedding(target))
        out = emb
        for layer in self.layers:
            out = layer(
                out, encoded, target_mask, None, target_pad_mask, encoded_pad_mask
            )
        return self.out(out)
