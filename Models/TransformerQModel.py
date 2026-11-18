import torch
import torch.nn as nn

from .checkpoint import Checkpoint

class TransformerQModel(Checkpoint, nn.Module):
    def __init__(self,
        input_size: int= 1,
        num_actions: int=1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.05,
        max_len: int = 64,
    ):
        """max_len bounds the sequence length T (the decisions of one player in one hand); the
        learned position embedding has max_len rows and forward raises beyond it."""
        super().__init__()
        self.config = dict(input_size=input_size, num_actions=num_actions, d_model=d_model, nhead=nhead,
                           num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, max_len=max_len)

        self.input_size = input_size
        self.d_model = d_model
        self.max_len = max_len

        self.in_proj = nn.Linear(input_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,     # <- key: avoids permute
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.out = nn.Linear(d_model, num_actions)

    def forward(self, x, lengths=None) -> torch.Tensor:
        # Accept:
        # 1D: (F,) -> (1, 1, F)
        # 2D: (B, F) -> (B, 1, F)
        # 3D: (B, T, F) -> ok
        # lengths: for a right-padded batch, each sequence's length; pad positions are masked out of
        # attention and the output is read at each sequence's last real token. Without it, the last token.
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"TransformerQModel expected 1D/2D/3D input, got {tuple(x.shape)}")
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len={self.max_len}; build the model with a larger max_len")

        x = self.in_proj(x) + self.pos(torch.arange(T, device=x.device))  # (B, T, d_model)
        if lengths is None:
            h = self.encoder(x)[:, -1, :]                                  # (B, d_model)
        else:
            lengths = lengths.to(x.device)
            pad = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]  # True at pad positions
            h = self.encoder(x, src_key_padding_mask=pad)[torch.arange(B, device=x.device), lengths - 1]
        return self.out(h)                                                 # (B, num_actions)
