import torch
import torch.nn as nn

from .checkpoint import Checkpoint

class MLPModel(Checkpoint, nn.Module):
    def __init__(self, input_features:int  = 27, output_features:int =9, hidden:tuple=(256,128), dropout:float =0.05) -> None:
        super().__init__()
        self.config = dict(input_features=input_features, output_features=output_features, hidden=tuple(hidden), dropout=dropout)
        layers = []
        prev = input_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, output_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)
