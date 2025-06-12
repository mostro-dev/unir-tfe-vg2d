# /game_agent/dqn/memory/replay_memory.py

import random
from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, capacity=2000):
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.memory)
