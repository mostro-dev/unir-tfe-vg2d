# game_agent/dqn/train_dqn.py

from game_agent.dqn.environment import GameEnvironment
from game_agent.dqn.dqn_agent import DQNAgent  # Lo implementaremos luego

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100


def train():
    env = GameEnvironment()
    # estado = 4 direcciones, acciones = 5 posibles
    agent = DQNAgent(state_dim=4, action_dim=5)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"[EPISODIO {episode}] Recompensa total: {total_reward:.2f}")

    agent.save("dqn_model.pth")


if __name__ == "__main__":
    train()
