import os
import torch
import torch.nn as nn

class TransformerQModel(nn.Module):
    def __init__(self,
        input_size: int= 1,
        num_actions: int=1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        self.in_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,     # <- key: avoids permute
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, num_actions)

    def forward(self, x) -> torch.Tensor:
        # Accept:
        # 1D: (F,) -> (1, 1, F)
        # 2D: (B, F) -> (B, 1, F)
        # 3D: (B, T, F) -> ok
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"TransformerQModel expected 1D/2D/3D input, got {tuple(x.shape)}")

        x = self.in_proj(x)              # (B, T, d_model)
        h = self.encoder(x)              # (B, T, d_model)
        h_last = h[:, -1, :]             # (B, d_model)
        q = self.out(h_last)             # (B, num_actions)
        return q
    
    def save(self, file_name: str = "model.pth"):
        """
        Save the model's state dictionary to disk.

        The model parameters are saved to the `./model` directory. If the
        directory does not exist, it is created automatically.

        Args:
            file_name (str): Name of the file to save the model state to.
                Defaults to "model.pth".
        """
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

        
    @classmethod
    def load(cls, file_name: str = "model.pth", device: str = "cpu"):
        """
        Load a model from a saved state dictionary.

        This method instantiates the model using the default constructor,
        loads the saved parameters, moves the model to the specified device,
        and sets it to evaluation mode.

        Args:
            file_name (str): Name of the saved model file located in the
                `./model` directory. Defaults to "model.pth".
            device (str): Device to load the model onto (e.g. "cpu", "cuda",
                "cuda:0"). Defaults to "cpu".

        Returns:
            nn.Module: The loaded model in evaluation mode.
        """
        file_path = os.path.join("./model", file_name)

        model = cls()  # calls __init__
        state_dict = torch.load(file_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model
