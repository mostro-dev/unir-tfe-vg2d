# game_agent/dqn/train_dqn.py

from game_agent.dqn.agent.dqn_agent import DQNAgent
from game_agent.dqn.environment import GameEnvironment

import numpy as np

NUM_EPISODES = 3
MAX_STEPS_PER_EPISODE = 10

ACTIONS = ['up', 'right', 'down', 'left', 'z']


def train():
    env = GameEnvironment(True)  # True para guardar el mapa
    agent = DQNAgent(state_dim=4, action_dim=len(ACTIONS))

    # Para llevar la recompensa de los últimos episodios
    episode_rewards = []

    global_step = 0

    for episode in range(NUM_EPISODES):
        state = env.get_state()
        total_reward = 0
        collisions = 0      # movimientos infructuosos
        interactions = 0    # pulsaciones de Z que abren diálogo

        for step in range(MAX_STEPS_PER_EPISODE):
            global_step += 1

            action = agent.select_action(state)
            action_idx = ACTIONS.index(action)

            next_state, reward, done = env.step(action)
            agent.store_transition(state, action_idx, reward, next_state, done)
            print(
                f"   [Paso {step+1}] Acción: {action}, Recompensa: {reward:.2f}")
            agent.train()

            total_reward += reward
            if action in ['up', 'right', 'down', 'left'] and reward < 0:
                collisions += 1
            if action == 'z' and reward > 0:
                interactions += 1

            state = next_state

            if global_step % 50 == 0:
                print(
                    f"[Paso {global_step}] Acción: {action}, Recompensa: {reward:.2f}")

            if done:
                break

        # Guardar recompensa y calcular métricas de resumen
        episode_rewards.append(total_reward)
        avg_recent_reward = np.mean(episode_rewards[-50:])  # media últimos 50
        collision_rate = collisions / step
        interaction_rate = interactions / step
        print(
            f"\n[EPISODIO {episode+1:3d}] "
            f"Rew: {total_reward:6.2f}  "
            f"Average Reward: {avg_recent_reward:6.2f}  "
            f"Colisiones: {collision_rate:.2f}  "
            f"Interac: {interaction_rate:.2f}"
        )

    # Aquí iremos añadiendo checkpoints más adelante

    agent.save("dqn_model.keras")


if __name__ == "__main__":
    train()
