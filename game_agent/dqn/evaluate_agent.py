from game_agent.dqn.environment import GameEnvironment
from game_agent.dqn.agent.dqn_agent import DQNAgent
import time


def evaluate(num_steps=50):
    print("[EVAL] Cargando entorno y modelo...")
    env = GameEnvironment()
    agent = DQNAgent(state_dim=4, action_dim=5)
    agent.load("dqn_model.keras")  # Asegúrate de que este archivo exista

    state = env.get_state()
    total_reward = 0

    print(f"[EVAL] Iniciando evaluación por {num_steps} pasos...")

    for step in range(num_steps):
        # Usamos solo la política aprendida
        action = agent.select_action(state, explore=False)
        next_state, reward, done = env.step(action)
        print(
            f"[EVAL] Paso {step + 1}: Acción={action}, Recompensa={reward:.2f}")
        total_reward += reward
        state = next_state

        time.sleep(2)  # Para observar mejor el comportamiento

        if done:
            print("[EVAL] Episodio terminado anticipadamente.")
            break

    print(
        f"[EVAL] Evaluación completada. Recompensa total: {total_reward:.2f}")


if __name__ == "__main__":
    evaluate()
