# main.py
from game_agent.dqn.evaluate_agent import evaluate
from config import GAME_REGION, TILE_HEIGHT, TILE_WIDTH
from game_agent.controller.keyboard_controller import move, press
from game_agent.controller.keyboard_controller import press, move
from game_agent.dqn.agent.dqn_agent import DQNAgent
from game_agent.dqn.environment import GameEnvironment
from game_agent.dqn.train_dqn import train
from game_agent.dqn.train.train_loop import run_training
from game_agent.vision.dialog_detector import is_dialog_open_by_template
from game_agent.vision.screen_reader import capture_region, save_image
from game_agent.vision.tile_utils import get_surrounding_obstacles, split_into_tiles
from game_agent.vision.transform_pipeline import save_image_pipeline
import numpy as np
import time


# Move test 1
print("Starting in 3 seconds...")
time.sleep(3)


if __name__ == "__main__":
    print("Iniciando entrenamiento del agente DQN...")
    train()


# if __name__ == "__main__":
#     evaluate(num_steps=50)

# Inicializar entorno

# game_env = GameEnvironment()
# player_top_left = game_env.player_pos

# frame = capture_region(GAME_REGION)

# if is_dialog_open_by_template(frame):
#     print("💬 Cuadro de texto detectado. Presionando Z.")
#     press('z')
#     time.sleep(0.5)


# def images_different(img1, img2, threshold=0.05):
#     diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32)) / 255.0
#     print(f"[DEBUG] Diferencia de imágenes: {np.mean(diff)}")
#     return np.mean(diff) >= threshold


# # Loop de movimiento tipo serpiente
# moving_right = True

# while True:
#     state = game_env.get_state()
#     obstacles = {
#         "up": bool(state[0]),
#         "right": bool(state[1]),
#         "down": bool(state[2]),
#         "left": bool(state[3]),
#     }

#     if moving_right:
#         if not obstacles["right"]:
#             print("➡️  Moviendo a la derecha")
#             move("right")
#             time.sleep(1)
#         else:
#             print("🚧 Posible obstáculo a la derecha")
#             if game_env.is_real_obstacle("right"):
#                 print("🧱 Confirmado: hay pared a la derecha")
#                 print("⬇️  Validando si se puede bajar")
#                 if not game_env.is_real_obstacle("down"):
#                     print("⬇️  Bajando")
#                     move("down")
#                     time.sleep(1)
#                     moving_right = False
#                 else:
#                     print("🔚 Fin del camino (no se puede bajar)")
#                     break
#     else:
#         if not obstacles["left"]:
#             print("⬅️  Moviendo a la izquierda")
#             move("left")
#             time.sleep(1)
#         else:
#             print("🚧 Posible obstáculo a la izquierda")
#             if game_env.is_real_obstacle("left"):
#                 print("🧱 Confirmado: hay pared a la izquierda")
#                 print("⬇️  Validando si se puede bajar")
#                 if not game_env.is_real_obstacle("down"):
#                     print("⬇️  Bajando")
#                     move("down")
#                     time.sleep(1)
#                     moving_right = True
#                 else:
#                     print("🔚 Fin del camino (no se puede bajar)")
#                     break

# LLM Command Execution
# while True:
#     cmd = input("¿Qué debe hacer el personaje? ('salir' para terminar): ")
#     if cmd.lower() == "salir":
#         break
#     try:
#         print("Waiting 3 seconds...")
#         time.sleep(3)
#         process_command_with_llm(cmd)
#     except Exception as e:
#         print(f"Error: {e}")

# Image Capture
# frame = capture_region(GAME_REGION)
# save_image(frame)
# pipeline_image = save_image_pipeline(frame)

# tiles = split_into_tiles(
#     pipeline_image, tile_height=TILE_HEIGHT, tile_width=TILE_WIDTH)
# obstacles = get_surrounding_obstacles(tiles, player_top_left=(3, 4), debug=False)

# print("Obstacles detected:", obstacles)


# env = GameEnvironment()
# state = env.get_state()
# print("Estado:", state)
