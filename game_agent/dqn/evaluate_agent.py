# game_agent/dqn/evaluate_agent.py

import time
from game_agent.dqn.environment import GameEnvironment
from game_agent.dqn.agent.dqn_agent import DQNAgent
# ['up','right','down','left','z']
from game_agent.dqn.constants import ACTIONS


def evaluate(num_steps=50):
    print("[EVAL] Cargando entorno y modelo…")
    env = GameEnvironment(save_mode=False, punish_revisit=False)
    env.world_map.load()  # cargamos el mapa previamente guardado

    agent = DQNAgent(state_dim=24, action_dim=len(ACTIONS))
    agent.load("dqn_model.keras")
    print("[EVAL] Modelo cargado desde dqn_model.keras")

    state = env.get_state()
    total_reward = 0.0

    # métricas
    collisions = 0
    interactions = 0
    unique_tiles_start = len(env.world_map.map)

    print(f"[EVAL] Iniciando evaluación por {num_steps} pasos…")
    for step in range(1, num_steps + 1):
        # 1) obtenemos la acción del agente
        action_idx = agent.select_action(state, explore=False)
        if isinstance(action_idx, int):
            action = ACTIONS[action_idx]
        else:
            action = action_idx

        # 2) ejecutamos step y recogemos recompensa
        next_state, reward, done = env.step(action)

        # 3) contabilizamos metrics básicas
        if action in ['up', 'right', 'down', 'left'] and reward < 0:
            collisions += 1
        if action == 'z' and reward > 0:
            interactions += 1

        total_reward += reward
        state = next_state

        print(
            f"[EVAL] Paso {step:2d}: Acción={action:<5}  Recompensa={reward:6.2f}")

        time.sleep(2)
        if done:
            print("[EVAL] Episodio terminado anticipadamente.")
            break

    unique_tiles_end = len(env.world_map.map)
    avg_reward = total_reward / step

    print("\n[EVAL] ===== Resultados =====")
    print(f" Pasos totales......: {step}")
    print(f" Recompensa total..: {total_reward:.2f}")
    print(f" Recompensa media..: {avg_reward:.2f}")
    print(f" Choques (walls)...: {collisions} ({collisions/step:.2%})")
    print(f" Interacciones (z).: {interactions} ({interactions/step:.2%})")
    print(f" Tiles únicos visit.: {unique_tiles_start} → {unique_tiles_end}")
    print("[EVAL] ======================\n")


if __name__ == "__main__":
    evaluate()
