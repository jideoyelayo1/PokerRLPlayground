import torch
import torch.nn as nn

from .checkpoint import Checkpoint

class GRUModel(Checkpoint, nn.Module):
    """
    Gated Recurrent Unit (GRU) neural network.

    Processes sequential input data and produces per-sequence outputs.
    """
    def __init__(self, input_dim:int =1, output_dim:int =1, hidden_dim:int =1024, layer_dim:int =3, dropout:float =0.05):
        """
        Gated Recurrent Unit (GRU) neural network model.

        This model consists of a multi-layer GRU followed by a fully
        connected linear layer to map hidden states to outputs.

        Args:
            input_dim (int): Number of input features per timestep.
            output_dim (int): Number of output features.
            hidden_dim (int): Number of hidden units in each GRU layer.
            layer_dim (int): Number of stacked GRU layers.
            dropout (float): Dropout probability applied between GRU layers
                (only used when layer_dim > 1).
        """
        super().__init__()
        self.config = dict(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, dropout=dropout)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout if layer_dim > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x:torch.Tensor, lengths=None):
        # lengths: for a right-padded batch (B, T, F), each sequence's length; the output is taken
        # at its last real step, as if each sequence were run alone
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"GRUModel expected 1D/2D/3D input, got {tuple(x.shape)}")

        if lengths is not None:
            # packed: each sequence runs only its own steps; the last layer's final hidden state is
            # its output at its last real step
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out = self.gru(packed)[1][-1]  # (B, H)
        else:
            out, _ = self.gru(x)       # (B, T, H)
            out = out[:, -1, :]        # (B, H)
        out = self.fc(out)         # (B, output_dim)
        return out
