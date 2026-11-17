import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class QTrainer:
    def __init__(self, model: nn.Module, target_model: nn.Module, lr: float, gamma: float, device="cpu"):
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    @staticmethod
    def _unwrap_logits(y):
        # handle models returning (logits, h, c) etc.
        if isinstance(y, (tuple, list)):
            return y[0]
        return y

    def _forward_q(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        x arrives as (B, F). Some models (LSTM/GRU/Transformer-with-seq) may want (B, T, F).
        We'll try (B, F) first; if that fails due to dim requirements, retry with (B, 1, F).
        Always return (B, N_ACTIONS).
        Args:
            net (nn.Module): The Model
            x (torch.Tensor): The tensor
        """
        try:
            y = net(x)
        except (RuntimeError, ValueError) as e:
            if x.dim() == 2:
                y = net(x.unsqueeze(1))  # (B, 1, F)
            else:
                raise e

        y = self._unwrap_logits(y)

        # If model returned (N_ACTIONS,) for single batch, expand back to (1, N_ACTIONS)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        return y  # (B, N_ACTIONS)

    def train_step(self, states, actions, rewards, next_states, dones, next_masks=None):
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)

        if states.dim() == 1:
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)

        pred = self.model(states)  # (B, N_ACTIONS)

        with torch.no_grad():
            next_q = self.target_model(next_states)       # (B, N_ACTIONS)
            if next_masks is not None:                    # max over the legal actions at s' only
                next_q = next_q + torch.tensor(np.array(next_masks), dtype=torch.float32, device=self.device)
            max_next_q = next_q.max(dim=1).values         # (B,)
            target_q = rewards + (~dones).float() * self.gamma * max_next_q

        target = pred.clone().detach()
        target[torch.arange(target.size(0), device=self.device), actions] = target_q

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
