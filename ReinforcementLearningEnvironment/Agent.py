import torch
import torch.nn as nn
import copy
import random
import numpy as np


from ReinforcementLearningEnvironment.QTrainer import QTrainer
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import MAX_MEMORY, BATCH_SIZE, LEARNING_RATE, N_ACTIONS


class DQNAgent:
    def __init__(self, player_id: int, model: nn.Module, state_dim: int, gamma=0.9, device="cpu",
                 memory_size=MAX_MEMORY, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                 eps_start=1.0, eps_end=0.05, eps_decay_hands=10_000):
        self.player_id = player_id
        self.model = model.to(device)
        self.gamma = gamma
        self.device = device

        self.n_hands = 0
        # Epsilon-greedy: the chance of a uniformly random legal action falls linearly from
        # eps_start to eps_end over the first eps_decay_hands hands, then stays at eps_end.
        self.eps_start, self.eps_end, self.eps_decay_hands = eps_start, eps_end, eps_decay_hands
        self.epsilon = eps_start

        # Replay buffer: a list used as a ring once full. random.sample on a deque indexes it in
        # O(n) per item (about 50 ms per 1,024-sample at a million transitions); a list is O(1).
        self.memory = []
        self.memory_size = memory_size
        self.memory_pos = 0
        
        self.target_model = copy.deepcopy(self.model).to(device)
        self.target_model.eval()
        self.batch_size = batch_size
        self.trainer = QTrainer(self.model, self.target_model, lr=lr, gamma=self.gamma, device=device)
        self.target_update_freq = 200


    def remember(self, s, a, r, s2, done, mask2=None):
        """mask2: legal mask at s2 (0 legal, -1e9 not), so the target maxes over legal actions only."""
        t = (s, a, r, s2, done, np.zeros(N_ACTIONS, np.float32) if mask2 is None else mask2)
        if len(self.memory) < self.memory_size:
            self.memory.append(t)
        else:
            self.memory[self.memory_pos] = t
            self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def train_long_memory(self):
        if len(self.memory) == 0:
            return
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = self.memory
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, next_masks)
        if self.n_hands % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())


    def select_action(self, state_vec: np.ndarray, legal_mask: np.ndarray) -> int:
        frac = min(1.0, self.n_hands / self.eps_decay_hands) if self.eps_decay_hands > 0 else 1.0
        self.epsilon = self.eps_start + (self.eps_end - self.eps_start) * frac

        if random.random() < self.epsilon:
            legal_actions = np.where(legal_mask == 0.0)[0].tolist()
            return random.choice(legal_actions)

        st = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
        if st.dim() == 1:
            st = st.unsqueeze(0)  # (1, F)

        with torch.no_grad():
            q = self.model(st)     # (1, N_ACTIONS) for all models if standardized
            if isinstance(q, (tuple, list)):
                q = q[0]
            if q.dim() == 2:
                q = q[0]           # (N_ACTIONS,)

            q = q + torch.tensor(legal_mask, dtype=torch.float32, device=self.device)
            return int(torch.argmax(q).item())
