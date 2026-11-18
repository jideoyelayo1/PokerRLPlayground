import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class QTrainer:
    def __init__(self, model: nn.Module, target_model: nn.Module, lr: float, gamma: float, device="cpu", double=False):
        self.gamma = gamma
        # Double DQN: the online network picks the next action, the target network values it
        self.double = double
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

    def _batch(self, xs):
        """States as a (B, F) tensor; sequences of states ((T_i, F) arrays) as (B, max T_i, F),
        right-padded with zeros, plus the lengths T_i."""
        if np.ndim(xs[0]) == 1:
            return torch.tensor(np.array(xs), dtype=torch.float32, device=self.device), None
        lengths = [len(x) for x in xs]
        out = np.zeros((len(xs), max(lengths), xs[0].shape[1]), dtype=np.float32)
        for i, x in enumerate(xs):
            out[i, :len(x)] = x
        return torch.tensor(out, device=self.device), torch.tensor(lengths, device=self.device)

    @staticmethod
    def _q(net, x, lengths):
        return net(x) if lengths is None else net(x, lengths)

    def train_step(self, states, actions, rewards, next_states, dones, next_masks=None):
        states, lengths = self._batch(states)
        next_states, next_lengths = self._batch(next_states)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)

        pred = self._q(self.model, states, lengths)  # (B, N_ACTIONS)

        with torch.no_grad():
            next_q = self._q(self.target_model, next_states, next_lengths)  # (B, N_ACTIONS)
            m = 0.0 if next_masks is None else torch.tensor(np.array(next_masks), dtype=torch.float32, device=self.device)
            if self.double:
                pick = (self._q(self.model, next_states, next_lengths) + m).argmax(dim=1, keepdim=True)
                max_next_q = next_q.gather(1, pick).squeeze(1)  # (B,)
            else:
                max_next_q = (next_q + m).max(dim=1).values  # (B,); max over the legal actions at s' only
            target_q = rewards + (~dones).float() * self.gamma * max_next_q

        target = pred.clone().detach()
        target[torch.arange(target.size(0), device=self.device), actions] = target_q

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
