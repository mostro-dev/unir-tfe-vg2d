# game_agent/dqn/train_dqn.py

from game_agent.dqn.agent.dqn_agent import DQNAgent
from game_agent.dqn.environment import GameEnvironment

import numpy as np

NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 60

ACTIONS = ['up', 'right', 'down', 'left', 'z']


def train():
    env = GameEnvironment()
    agent = DQNAgent(state_dim=4, action_dim=len(ACTIONS))

    global_step = 0

    for episode in range(NUM_EPISODES):
        state = env.get_state()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            global_step += 1

            action = agent.select_action(state)
            action_idx = ACTIONS.index(action)

            next_state, reward, done = env.step(action)
            agent.store_transition(state, action_idx, reward, next_state, done)
            print(
                f"   [Paso {step+1}] Acción: {action}, Recompensa: {reward:.2f}")
            agent.train()

            state = next_state
            total_reward += reward

            if global_step % 50 == 0:
                print(
                    f"[Paso {global_step}] Acción: {action}, Recompensa: {reward:.2f}")

            if done:
                break

        print(f"[EPISODIO {episode+1}] Total recompensa: {total_reward:.2f}")

    agent.save("dqn_model.keras")


if __name__ == "__main__":
    train()
