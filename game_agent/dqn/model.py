import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_dim, action_dim, gamma=0.99, lr=1e-3, buffer_size=10000, batch_size=64):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = DQN(input_dim, action_dim)
        self.target_net = DQN(input_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)

        q_values = self.policy_net(state).gather(
            1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
