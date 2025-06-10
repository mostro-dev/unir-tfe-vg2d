from config import GAME_REGION, TILE_HEIGHT, TILE_WIDTH
from game_agent.controller.keyboard_controller import move, press
from game_agent.vision.screen_reader import capture_region, save_image
from game_agent.vision.transform_pipeline import save_image_pipeline
from game_agent.vision.tile_utils import split_into_tiles, get_surrounding_obstacles
import numpy as np
import time


class GameEnvironment:
    def __init__(self):
        self.tile_height = TILE_HEIGHT
        self.tile_width = TILE_WIDTH
        self.player_pos = (4, 3)  # tile superior izquierdo del jugador

    def capture_and_process(self):
        frame = capture_region(GAME_REGION)
        save_image(frame)
        pipeline_image = save_image_pipeline(frame)
        return pipeline_image

    def get_state(self):
        processed_image = self.capture_and_process()
        tiles = split_into_tiles(
            processed_image, self.tile_height, self.tile_width)
        obstacles = get_surrounding_obstacles(
            tiles, player_top_left=self.player_pos)

        state = np.array([
            int(obstacles["up"]),
            int(obstacles["right"]),
            int(obstacles["down"]),
            int(obstacles["left"]),
        ], dtype=np.float32)

        return state

    def image_changed(self, img1, img2, threshold=15):
        """Compara el promedio absoluto de diferencia entre dos imágenes."""
        if img1.shape != img2.shape:
            return True
        diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
        mean_diff = np.mean(diff)
        return mean_diff > threshold

    def step(self, action):
        prev_image = self.capture_and_process()

        # Ejecutar acción
        if action in ['up', 'right', 'down', 'left',]:
            move(action)
        elif action == 'z':
            press('z')

        new_image = self.capture_and_process()
        moved = self.image_changed(prev_image, new_image)

        if action in ['up', 'right', 'down', 'left', ]:
            reward = +1 if moved else -1
        elif action == 'z':
            reward = 0.5 if moved else -0.5

        state = self.get_state()
        done = False  # Podrías definir condiciones de finalización más adelante

        return state, reward, done
