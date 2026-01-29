import os
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_features:int  = 27, output_features:int =9, hidden:tuple=(256,128), dropout:float =0.05) -> None:
        super().__init__()
        layers = []
        prev = input_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, output_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
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

        