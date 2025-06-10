import time
from game_agent.dqn.environment import GameEnvironment
from game_agent.controller.keyboard_controller import move
from game_agent.vision.tile_utils import get_surrounding_obstacles

# [DEPRECATED]

env = GameEnvironment()
player_top_left = env.player_pos


# Función de ayuda para verificar si un obstáculo es real
def is_real_obstacle(direction):
    move(direction)
    time.sleep(0.8)  # espera para ver si realmente se movió
    new_state = env.get_state()
    return new_state[["up", "down", "left", "right"].index(direction)] == 1


# Movimiento serpenteante: derecha hasta tope, luego abajo, luego izquierda
moving_right = True
while True:
    state = env.get_state()
    obstacles = {
        "right": bool(state[3]),
        "left": bool(state[2]),
        "down": bool(state[1]),
    }

    if moving_right:
        if not obstacles["right"]:
            move("right")
            time.sleep(0.5)
        else:
            if is_real_obstacle("right"):
                print("Pared detectada a la derecha. Intentando bajar.")
                if not obstacles["down"]:
                    move("down")
                    time.sleep(0.5)
                    moving_right = False
                else:
                    print("Fin del camino.")
                    break
    else:  # moving left
        if not obstacles["left"]:
            move("left")
            time.sleep(0.5)
        else:
            if is_real_obstacle("left"):
                print("Pared detectada a la izquierda. Intentando bajar.")
                if not obstacles["down"]:
                    move("down")
                    time.sleep(0.5)
                    moving_right = True
                else:
                    print("Fin del camino.")
                    break
