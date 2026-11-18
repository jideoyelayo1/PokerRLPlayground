import torch
import torch.nn as nn

from .checkpoint import Checkpoint

class LSTMModel(Checkpoint, nn.Module):
    def __init__(self, input_dim:int =1, output_dim:int =1, hidden_dim:int =1024, layer_dim:int =3, dropout:float =0.05):
        super().__init__()
        self.config = dict(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, dropout=dropout)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout if layer_dim > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths=None):
        # lengths: as in GRUModel.forward, for right-padded batches
        # Accepts:
        # 1D: (F,) -> (1, 1, F)
        # 2D: (B, F) -> (B, 1, F)
        # 3D: (B, T, F) -> (B, T, F)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"LSTMModel expected 1D/2D/3D input, got {tuple(x.shape)}")

        if lengths is not None:  # packed, as in GRUModel: final hidden state of the last layer
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out = self.lstm(packed)[1][0][-1]  # (B, H)
        else:
            out, _ = self.lstm(x)          # out: (B, T, H)
            out = out[:, -1, :]            # last timestep: (B, H)
        out = self.fc(out)             # (B, output_dim)
        return out
