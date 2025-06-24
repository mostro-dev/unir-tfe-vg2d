# game_agent/dqn/dqn_agent.py

import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    ACTIONS = ['up', 'right', 'down', 'left', 'z']

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hiperparámetros
        self.gamma = 0.95
        self.epsilon = 1.0  # exploración
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def select_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return random.choice(self.ACTIONS)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return self.ACTIONS[np.argmax(q_values[0])]

    def select_action_old(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * \
                    np.amax(self.model.predict(
                        np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)[0]
            target_f[action] = target
            states.append(state)
            targets.append(target_f)

        self.model.fit(np.array(states), np.array(
            targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save(path)
        print(f"[DEBUG] Modelo guardado en {path}")

    def load(self, path):
        self.model = load_model(path)
        print(f"[DEBUG] Modelo cargado desde {path}")
