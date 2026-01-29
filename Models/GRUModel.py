import os
import torch
import torch.nn as nn

class GRUModel(nn.Module):
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
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout if layer_dim > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x:torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"GRUModel expected 1D/2D/3D input, got {tuple(x.shape)}")

        out, _ = self.gru(x)       # (B, T, H)
        out = out[:, -1, :]        # (B, H)
        out = self.fc(out)         # (B, output_dim)
        return out
    
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
