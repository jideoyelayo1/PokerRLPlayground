import torch
import torch.nn as nn
from collections import deque
import copy
import random
import numpy as np


from ReinforcementLearningEnvironment.QTrainer import QTrainer
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import MAX_MEMORY, BATCH_SIZE, LEARNING_RATE


class DQNAgent:
    def __init__(self, player_id: int, model: nn.Module, state_dim: int, gamma=0.9, random_range=80, device="cpu"):
        self.player_id = player_id
        self.model = model.to(device)
        self.gamma = gamma
        self.device = device

        self.n_hands = 0
        self.random_range = random_range
        self.epsilon = 0

        self.memory = deque(maxlen=MAX_MEMORY)
        
        self.target_model = copy.deepcopy(self.model).to(device)
        self.target_model.eval()
        self.trainer = QTrainer(self.model, self.target_model, lr=LEARNING_RATE, gamma=self.gamma, device=device)
        self.target_update_freq = 200


    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_long_memory(self):
        if len(self.memory) == 0:
            return
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = list(self.memory)
        filtered = [
            (s, a, r, s2, d)
            for (s, a, r, s2, d) in batch
            if s is not None and s2 is not None
        ]

        if not filtered:
            return

        states, actions, rewards, next_states, dones = zip(*filtered)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        if self.n_hands % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())


    def train_short_memory(self, s, a, r, s2, done):
        self.trainer.train_step([s], [a], [r], [s2], [done])

    def select_action(self, state_vec: np.ndarray, legal_mask: np.ndarray) -> int:
        self.epsilon = max(0, self.random_range - self.n_hands)

        if random.randint(0, 200) < self.epsilon:
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
