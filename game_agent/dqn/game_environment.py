# game_agent/dqn/game_environment.py
import pyautogui
from game_agent.vision.tile_utils import split_into_tiles, get_surrounding_obstacles
from game_agent.vision.transform_pipeline import save_image_pipeline
import numpy as np

# [DEPRECATED]


class GameEnvironment:
    def __init__(self):
        self.position = (4, 3)  # Posición inicial del jugador en tiles
        self.visited = set()
        self.discovered_map = {}

    def reset(self):
        self.position = (4, 3)
        self.visited = set()
        self.discovered_map = {}
        return self._get_state()

    def _get_state(self):
        # Aquí se llamaría a la visión artificial para extraer tiles cercanos
        # y codificar la información de obstáculos
        from game_agent.vision.tile_utils import get_surrounding_obstacles
        tiles = self.get_current_tiles()  # <- función que procesaría el screenshot
        obstacles = get_surrounding_obstacles(
            tiles, player_top_left=self.position)
        return np.array(list(obstacles.values()), dtype=np.float32)

    def step(self, action):
        """
        Acción: 0 = up, 1 = down, 2 = left, 3 = right, 4 = z
        """
        actions = ['up', 'right', 'down', 'left', 'z']
        direction = actions[action]
        new_pos = self._calculate_new_position(direction)

        hit_obstacle = self.check_if_obstacle(new_pos)

        if direction == 'z':
            reward = 0.1  # Interacción ligera por ahora
        elif hit_obstacle:
            reward = -1.0
        else:
            self.position = new_pos
            reward = 0.2 if new_pos not in self.visited else -0.1
            self.visited.add(new_pos)

        done = False  # No hay condición de final todavía
        state = self._get_state()
        return state, reward, done, {}

    def _calculate_new_position(self, direction):
        r, c = self.position
        moves = {
            'up': (c, r - 1),
            'right': (c + 1, r),
            'down': (c, r + 1),
            'left': (c - 1, r),
            'z': (r, c)  # no cambia posición
        }
        return moves[direction]

    def check_if_obstacle(self, new_pos):
        # Procesar una nueva imagen después del movimiento y verificar si se movió
        # Podría ser comparando el centro del personaje entre frames
        # o usando el resultado de get_surrounding_obstacles
        return False  # Placeholder
