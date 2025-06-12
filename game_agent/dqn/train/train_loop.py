import time
import numpy as np
from game_agent.dqn.environment import GameEnvironment
from game_agent.dqn.agent.dqn_agent import DQNAgent

# Acciones disponibles para el agente
ACTIONS = ['up', 'right', 'down', 'left', 'z']

# Configuración
EPISODES = 100
MAX_STEPS = 50
BATCH_SIZE = 32


def run_training():
    env = GameEnvironment()
    agent = DQNAgent(state_size=4, action_size=len(ACTIONS))

    for episode in range(EPISODES):
        print(f"\n🎮 Episodio {episode + 1}")
        state = env.get_state()

        for step in range(MAX_STEPS):
            action_idx = agent.act(state)
            action = ACTIONS[action_idx]

            print(f"[{step}] Acción: {action}")
            next_state, reward, done = env.step(action)

            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state

            agent.replay(BATCH_SIZE)

            if done:
                print("🛑 Fin del episodio.")
                break

        # Al finalizar cada episodio puedes guardar el modelo si deseas
        # agent.model.save("dqn_model.h5")


if __name__ == "__main__":
    print("⌛ Comenzando entrenamiento...")
    time.sleep(3)
    run_training()
