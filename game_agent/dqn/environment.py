import cv2
from config import GAME_REGION, TILE_HEIGHT, TILE_WIDTH
from game_agent.controller.keyboard_controller import move, press
from game_agent.vision.oak_zone_detector import is_oak_zone_triggered
from game_agent.vision.screen_reader import capture_region, save_image
from game_agent.vision.transform_pipeline import save_image_pipeline
from game_agent.vision.tile_utils import split_into_tiles, get_surrounding_obstacles
import numpy as np
import time
from game_agent.vision.dialog_detector import is_dialog_open_by_template
from datetime import datetime
import os


class GameEnvironment:
    BUILDING_THRESHOLD = 85  # Umbral para detectar cambios drásticos al entrar a un edificio
    # Ruta de las plantillas de OAK
    OAK_TEMPLATE_PATH = "./game_agent/vision/templates/"

    def __init__(self):
        self.tile_height = TILE_HEIGHT
        self.tile_width = TILE_WIDTH
        self.player_pos = (4, 3)  # tile superior izquierdo del jugador
        self.is_text_in_screen = False

    def is_near_oak_zone(self, threshold=0.9):
        current_frame = self.capture_and_process()
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        for template_path in glob.glob(os.path.join(self.OAK_TEMPLATE_PATH, "oak_template_*.png")):
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue

            result = cv2.matchTemplate(
                current_gray, template, cv2.TM_CCOEFF_NORMED)
            max_val = np.max(result)
            if max_val >= threshold:
                print(
                    f"[OAK DETECTION] Coincidencia con {os.path.basename(template_path)}: {max_val:.2f}")
                return True

        return False

    def images_different(self, img1, img2, threshold=0.05, debug=False):
        diff = np.abs(img1.astype(np.float32) -
                      img2.astype(np.float32)) / 255.0
        if debug:
            print(f"[DEBUG] Diferencia de imágenes: {np.mean(diff)}")
        return np.mean(diff) >= threshold

    def save_tile_image(self, tile, label):
        """
        Guarda un tile como imagen con una etiqueta dada ('WALL', 'INFO').
        """

        folder = os.path.join("game_agent", "tiles", label)
        os.makedirs(folder, exist_ok=True)

        filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        path = os.path.join(folder, filename)

        cv2.imwrite(path, tile)
        print(f"[DEBUG] Tile guardado en {path}")

    def extract_tile_in_direction(self, direction):
        """
        Devuelve una imagen 2x2 de tiles en la dirección dada desde el jugador,
        considerando que player_pos = (col, row).
        """
        full_image = self.capture_and_process()
        tiles = split_into_tiles(full_image, self.tile_height, self.tile_width)
        c, r = self.player_pos

        dir_map = {
            "up":    [(c, r-2), (c+1, r-2), (c, r-1), (c+1, r-1)],
            "down":  [(c, r+2), (c+1, r+2), (c, r+3), (c+1, r+3)],
            "left":  [(c-2, r), (c-2, r+1), (c-1, r), (c-1, r+1)],
            "right": [(c+2, r), (c+2, r+1), (c+3, r), (c+3, r+1)],
        }

        positions = dir_map[direction]
        tile_images = []

        for col, row in positions:
            if 0 <= row < len(tiles) and 0 <= col < len(tiles[0]):
                tile_images.append(tiles[row][col])
            else:
                tile_images.append(np.zeros_like(tiles[0][0]))  # Padding negro

        top = np.hstack(tile_images[:2])
        bottom = np.hstack(tile_images[2:])
        return np.vstack([top, bottom])

    def is_real_obstacle(self, direction, threshold=0.05, debug=True):
        before_move = self.capture_and_process()
        move(direction)
        time.sleep(1)

        after_move = self.capture_and_process()
        if self.images_different(before_move, after_move, threshold):
            if debug:
                print(f"[DEBUG] Se movió exitosamente hacia '{direction}'.")
            return False

        press('z')
        time.sleep(1)
        while is_dialog_open_by_template(capture_region(GAME_REGION)):
            print("💬 Cuadro de texto detectado. Presionando Z.")
            press('z')
            time.sleep(1)

        move(direction)
        time.sleep(1)

        final = self.capture_and_process()

        if self.images_different(after_move, final, threshold):
            if debug:
                print(
                    f"[DEBUG] Se movió después de interactuar con '{direction}'.")
            # Guarda el tile como INFO
            tile = self.extract_tile_in_direction(direction)
            self.save_tile_image(tile, label="INFO")
            return False

        if debug:
            print(f"[DEBUG] Obstáculo real en dirección '{direction}'.")
            tile = self.extract_tile_in_direction(direction)
            self.save_tile_image(tile, label="WALL")
            return True

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

    def image_changed(self, img1, img2, threshold=15, debug=True):
        """Compara el promedio absoluto de diferencia entre dos imágenes."""
        if img1.shape != img2.shape:
            return True
        diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
        mean_diff = np.mean(diff)
        if debug:
            print(f"[DEBUG] Diferencia media entre imágenes: {mean_diff}")
        return mean_diff > threshold

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

    def handle_oak_zone(self, current_image):
        """
        Detecta si el jugador está en una zona que activa el evento del Profesor Oak.
        Si lo está, lo fuerza a moverse hacia abajo y penaliza con una recompensa negativa.
        """
        if is_oak_zone_triggered(current_image, debug=False):
            print(
                "⚠️ Zona peligrosa del Profesor Oak detectada. Forzando salida hacia abajo...")
            for _ in range(5):
                move("down")
                time.sleep(1)
            return True, -10.0  # Indica que fue interceptado y aplica penalización
        return False, 0.0

    def step(self, action):
        prev_image = self.capture_and_process()

        # Detectar si estamos en zona de evento del Profesor Oak
        triggered, oak_penalty = self.handle_oak_zone(prev_image)
        if triggered:
            state = self.get_state()
            return state, oak_penalty, False

        if action in ['up', 'right', 'down', 'left']:
            move(action)
            self.last_direction = action
            time.sleep(1)
        elif action == 'z':
            press('z')
            time.sleep(1)

        new_image = self.capture_and_process()
        moved = self.image_changed(prev_image, new_image, threshold=15)
        print(f"[DEBUG] Movimiento detectado: {moved}")

        reward = 0

        if action in ['up', 'right', 'down', 'left']:
            if moved:
                reward += 0.2
                print(
                    f"[DEBUG] Se movió correctamente con {action}. Recompensa: +0.2")
            else:
                print(
                    f"[DEBUG] No se detectó movimiento con {action}. Verificando si es obstáculo real...")
                is_wall = self.is_real_obstacle(
                    action, threshold=5, debug=True)

                if is_wall:
                    reward -= 1.0
                    print(
                        f"[DEBUG] Confirmado obstáculo con {action}. Recompensa: -1.0")
                else:
                    reward -= 0.5
                    print(
                        f"[DEBUG] No fue obstáculo, quizá delay o animación. Recompensa: -0.5")

        elif action == 'z':
            frame = capture_region(GAME_REGION)
            is_dialog = is_dialog_open_by_template(frame)

            if is_dialog and not self.is_text_in_screen:
                reward += 2.0
                self.is_text_in_screen = True
                print("[DEBUG] Interacción exitosa con Z. Recompensa: +2.0")

                if hasattr(self, 'last_direction'):
                    tile = self.extract_tile_in_direction(self.last_direction)
                    self.save_tile_image(tile, label="INFO")

            elif is_dialog and self.is_text_in_screen:
                press('z')
                time.sleep(0.5)
            elif not is_dialog and self.is_text_in_screen:
                self.is_text_in_screen = False

        if self.image_changed(prev_image, new_image, threshold=self.BUILDING_THRESHOLD):
            print("🏠 Cambio visual fuerte detectado, probablemente entró a un edificio")
            reward += 3.0

            for _ in range(10):
                move("down")
                time.sleep(1)
                after_exit = self.capture_and_process()
                if not self.image_changed(prev_image, after_exit, threshold=self.BUILDING_THRESHOLD):
                    print("🚪 Salió del edificio.")
                    door_tile = self.extract_tile_in_direction("up")
                    self.save_tile_image(door_tile, label="DOOR")
                    reward += 0.5
                    break
            else:
                print("❌ No logró salir del edificio")
                reward -= 10.0

        state = self.get_state()
        done = False

        return state, reward, done
