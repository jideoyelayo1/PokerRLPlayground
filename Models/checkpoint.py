import os
import torch


class Checkpoint:
    """save/load shared by the models. A model sets self.config to its constructor kwargs;
    the file holds {"config", "state_dict"} so load() can rebuild the same architecture."""

    def save(self, file_name: str = "model.pth"):
        """Write config and weights to ./model/<file_name> (the directory is created if missing)."""
        os.makedirs("./model", exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict()}, os.path.join("./model", file_name))

    @classmethod
    def load(cls, file_name: str = "model.pth", device: str = "cpu", **kwargs):
        """Rebuild the model from ./model/<file_name> and return it on `device` in eval mode.
        kwargs override the saved config. A legacy file holding a bare state_dict has no config,
        so it loads only when the constructor kwargs are given."""
        # weights_only: the config is ints, floats and tuples
        ckpt = torch.load(os.path.join("./model", file_name), map_location=device, weights_only=True)
        if set(ckpt) == {"config", "state_dict"}:
            config, state_dict = ckpt["config"], ckpt["state_dict"]
        elif kwargs:
            config, state_dict = {}, ckpt
        else:
            raise ValueError(f"{file_name} is a bare state_dict without its architecture: "
                             f"pass the constructor kwargs, e.g. {cls.__name__}.load(..., **kwargs)")
        model = cls(**{**config, **kwargs})
        model.load_state_dict(state_dict)
        return model.to(device).eval()
