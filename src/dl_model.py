"""Part C — GRU encoder-decoder model.

A single global network shared across all store-SKU series (multi-series modelling).

    encoder GRU  : reads the past `L`-day sequence  -> summary hidden state
    static MLP   : embeds categoricals + numerics   -> one static vector per series
    decoder GRU  : consumes known-future inputs (+ static), initialised from the
                   encoder state, and emits all `H` horizons at once (direct forecast)

Predicting all 14 days in one shot (rather than feeding predictions back in) avoids
the autoregressive error snowball and mirrors the classical baseline's direct setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoderDecoder(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        n_enc_feat: int,
        n_fut_feat: int,
        n_stat_num: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(card, embedding_dim) for card in cat_cardinalities
        )
        stat_in = len(cat_cardinalities) * embedding_dim + n_stat_num
        self.static_mlp = nn.Sequential(
            nn.Linear(stat_in, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            n_enc_feat, hidden_size, num_layers,
            batch_first=True, dropout=gru_dropout,
        )
        self.decoder = nn.GRU(
            n_fut_feat + hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=gru_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        enc: torch.Tensor,   # [B, L, n_enc_feat]
        fut: torch.Tensor,   # [B, H, n_fut_feat]
        scat: torch.Tensor,  # [B, n_cat]
        snum: torch.Tensor,  # [B, n_stat_num]
    ) -> torch.Tensor:       # [B, H]
        embs = [emb(scat[:, i]) for i, emb in enumerate(self.embeddings)]
        static = self.static_mlp(torch.cat(embs + [snum], dim=-1))  # [B, hidden]

        _, h_n = self.encoder(enc)  # h_n: [num_layers, B, hidden]

        H = fut.size(1)
        static_seq = static.unsqueeze(1).expand(-1, H, -1)          # [B, H, hidden]
        dec_in = torch.cat([fut, static_seq], dim=-1)
        dec_out, _ = self.decoder(dec_in, h_n)                      # [B, H, hidden]
        return self.head(dec_out).squeeze(-1)                       # [B, H]
