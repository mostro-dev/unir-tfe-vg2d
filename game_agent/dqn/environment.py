# game_agent/dqn/environment.py

import glob
import os
import time
import cv2
import numpy as np
from datetime import datetime

from config import GAME_REGION, TILE_HEIGHT, TILE_WIDTH, TileType
from game_agent.map.world_map import WorldMap
from game_agent.controller.keyboard_controller import move, press
from game_agent.vision.screen_reader import capture_region, save_image
from game_agent.vision.transform_pipeline import save_image_pipeline
from game_agent.vision.tile_utils import split_into_tiles, get_surrounding_obstacles
from game_agent.vision.dialog_detector import is_dialog_open_by_template
from game_agent.vision.oak_zone_detector import is_oak_zone_triggered


class GameEnvironment:
    BUILDING_THRESHOLD = 85  # umbral para detectar entrada a edificio
    OAK_STEPS_BACK = 2  # pasos hacia atrás en zona Oak

    REWARDS = {
        "move_success": 1,
        "move_revisit": 0,
        "move_wall": -1.0,
        "move_no_wall": -0.5,
        "interaction_success": 1.5,
        "building_entry": 2,
        "building_exit": 0.5,
        "oak_zone_penalty": -5
    }

    def __init__(self, save_mode: bool = True):
        # flag que indica si persistimos el world_map en disco
        self.save_mode = save_mode

        # visión
        self.tile_height = TILE_HEIGHT
        self.tile_width = TILE_WIDTH

        # mapa y posición lógica del agente
        self.world_map = WorldMap()
        self.agent_pos = (0, 0)
        # inicializamos la posición de arranque como piso
        self.world_map.update_tile(self.agent_pos, TileType.FLOOR)
        self.world_map.mark_visited(self.agent_pos)
        if self.save_mode:
            self.world_map.save()

        # estado para diálogo
        self.is_text_in_screen = False

    def _future_coord(self, action: str):
        """Devuelve la coordenada lógica resultante de aplicar `action`."""
        x, y = self.agent_pos
        if action == 'up':
            y -= 1
        elif action == 'down':
            y += 1
        elif action == 'left':
            x -= 1
        elif action == 'right':
            x += 1
        return (x, y)

    def capture_and_process(self):
        """Captura pantalla, guarda debug y pipeline, y devuelve la imagen procesada."""
        frame = capture_region(GAME_REGION)
        save_image(frame)
        return save_image_pipeline(frame)

    def handle_oak_zone(self, current_image):
        """
        Si la zona de Oak está activa, fuerza a bajar self.OAK_STEPS_BACK tiles
        y penaliza, actualizando también la posición lógica y el mapa.
        """
        if is_oak_zone_triggered(current_image, debug=False):
            print("⚠️ Zona Oak detectada: forzando salida ↓↓")
            for _ in range(self.OAK_STEPS_BACK):  # por ejemplo 2
                move("down")
                time.sleep(1)
                # *** actualizo la posición lógica y el mapa ***
                self.agent_pos = self._future_coord('down')
                self.world_map.update_tile(
                    self.agent_pos, TileType.FLOOR)
                self.world_map.mark_visited(self.agent_pos)
                if self.save_mode:
                    self.world_map.save()
            return True, -10.0
        return False, 0.0

    def image_changed(self, img1, img2, threshold=15, debug=True):
        """Compara diferencia media absoluta entre imágenes."""
        if img1.shape != img2.shape:
            return True
        diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
        mean_diff = np.mean(diff)
        if debug:
            print(f"[DEBUG] Difum promedio: {mean_diff:.2f}")
        return mean_diff > threshold

    def extract_tile_in_direction(self, direction):
        """Extrae el bloque de 2×2 tiles en `direction` relativas a agent_pos."""
        full = self.capture_and_process()
        tiles = split_into_tiles(full, self.tile_height, self.tile_width)
        cx, cy = self.agent_pos
        dir_map = {
            "up":    [(cx, cy-2), (cx+1, cy-2), (cx, cy-1), (cx+1, cy-1)],
            "down":  [(cx, cy+2), (cx+1, cy+2), (cx, cy+3), (cx+1, cy+3)],
            "left":  [(cx-2, cy), (cx-2, cy+1), (cx-1, cy), (cx-1, cy+1)],
            "right": [(cx+2, cy), (cx+2, cy+1), (cx+3, cy), (cx+3, cy+1)],
        }
        imgs = []
        for tx, ty in dir_map[direction]:
            if 0 <= ty < len(tiles) and 0 <= tx < len(tiles[0]):
                imgs.append(tiles[ty][tx])
            else:
                imgs.append(np.zeros_like(tiles[0][0]))
        top = np.hstack(imgs[:2])
        bot = np.hstack(imgs[2:])
        return np.vstack([top, bot])

    def is_real_obstacle(self, direction, threshold=0.05, debug=True):
        """
        Comprueba si tras intentar moverse e interactuar sigue bloqueado:
        si es pared → devuelve True (guarda WALL),
        si no → False (guarda INFO).
        """
        if debug:
            print(f"[DEBUG] Comprobando obstáculo en {direction}")
        before = self.capture_and_process()
        move(direction)
        time.sleep(1)
        after = self.capture_and_process()

        # Si se movió, no era obstáculo
        if self.image_changed(before, after, threshold):
            if debug:
                print(f"[DEBUG] Movió con éxito a {direction}")
            return False

        # Intentar interactuar
        press('z')
        time.sleep(1)
        while is_dialog_open_by_template(capture_region(GAME_REGION)):
            press('z')
            time.sleep(0.5)

        # Volver a mover
        move(direction)
        time.sleep(1)
        final = self.capture_and_process()

        # Si ahora sí se movió → INFO
        if self.image_changed(after, final, threshold):
            if debug:
                print(f"[DEBUG] Interactuó y luego movió a {direction}")
            coord = self._future_coord(direction)
            self.world_map.update_tile(coord, TileType.INFO)
            if self.save_mode:
                self.world_map.save()
            return False

        # Sino → WALL
        if debug:
            print(f"[DEBUG] Obstáculo real en {direction}")
        coord = self._future_coord(direction)
        self.world_map.update_tile(coord, TileType.WALL)
        if self.save_mode:
            self.world_map.save()
        return True

    def get_state(self):
        """
        Estado = vector [up, right, down, left]
        según get_surrounding_obstacles sobre la imagen procesada.
        """
        proc = self.capture_and_process()
        tiles = split_into_tiles(proc, self.tile_height, self.tile_width)
        obs = get_surrounding_obstacles(tiles, player_top_left=(4, 3))
        return np.array([
            float(obs["up"]), float(obs["right"]),
            float(obs["down"]), float(obs["left"])
        ], dtype=np.float32)

    def step(self, action: str, debug: bool = True):
        """
        Ejecuta la acción, calcula recompensa, actualiza posición lógica,
        y registra el tile en world_map (con penalización por revisitas).
        """
        # 1) captura previa y oak-zone
        prev_img = self.capture_and_process()
        triggered, oak_penalty = self.handle_oak_zone(prev_img)
        if triggered:
            # devolvemos el estado tras haber forzado la salida
            return self.get_state(), oak_penalty, False

        # 2) calculamos la coord lógica futura
        next_coord = self._future_coord(action)

        # 3) Ejecutamos la acción física
        if action in ['up', 'down', 'left', 'right']:
            move(action)
            self.last_direction = action
            time.sleep(1)
        else:  # acción 'z'
            press('z')
            time.sleep(1)

        # 4) Captura tras la acción y comparamos
        new_img = self.capture_and_process()
        moved = self.image_changed(prev_img, new_img, threshold=15)
        reward = 0.0

        # 5) Movimiento: recompensa o castigo, y actualización de visitas
        if action in ['up', 'down', 'left', 'right']:
            if moved:
                # Antes de marcar, leemos cuántas veces visitamos next_coord
                prev_visits = self.world_map.map.get(
                    next_coord, {}).get("_visits", 0)
                if prev_visits > 0:
                    reward += self.REWARDS["move_revisit"]
                    if debug:
                        print(
                            f"[DEBUG] Revisitando {next_coord}. Penalización: {self.REWARDS['move_revisit']}")
                else:
                    reward += self.REWARDS["move_success"]
                    if debug:
                        print(
                            f"[DEBUG] Primera visita a {next_coord}. Recompensa: {self.REWARDS['move_success']}")
                    self.world_map.update_tile(
                        next_coord, TileType.FLOOR)  # Marcamos como FLOOR

                # Actualizamos posición y marcamos visita
                self.agent_pos = next_coord
                self.world_map.mark_visited(self.agent_pos)
                if self.save_mode:
                    self.world_map.save()

            else:
                # No se movió: confirmamos si era pared real o no
                if self.is_real_obstacle(action):
                    reward += self.REWARDS["move_wall"]
                    if debug:
                        print(
                            f"[DEBUG] Choque contra pared con {action}. Recompensa: {self.REWARDS['move_wall']}")
                else:
                    reward += self.REWARDS["move_no_wall"]
                    if debug:
                        print(
                            f"[DEBUG] Sin movimiento pero no pared. Recompensa: {self.REWARDS['move_no_wall']}")

        # 6) Interacción con 'z'
        else:
            frame = capture_region(GAME_REGION)
            dialog = is_dialog_open_by_template(frame)
            if dialog and not self.is_text_in_screen:
                reward += self.REWARDS["interaction_success"]
                self.is_text_in_screen = True
                if debug:
                    print(
                        f"[DEBUG] Interacción exitosa con Z. Recompensa: {self.REWARDS['interaction_success']}")
                # Registramos INFO en la casilla con la última dirección
                coord = self._future_coord(self.last_direction)
                self.world_map.update_tile(coord, TileType.INFO)
                if self.save_mode:
                    self.world_map.save()

                # Mantenemos pulsando Z hasta que el diálogo desaparezca
                while True:
                    press('z')
                    time.sleep(1)
                    frame = capture_region(GAME_REGION)
                    if not is_dialog_open_by_template(frame):
                        if debug:
                            print("[DEBUG] Diálogo cerrado.")
                        self.is_text_in_screen = False
                        break

            elif not dialog and self.is_text_in_screen:
                # Se cerró el diálogo
                self.is_text_in_screen = False

        # 7) Detección de entrada a edificio
        if self.image_changed(prev_img, new_img, threshold=self.BUILDING_THRESHOLD):
            print("🏠 Cambio visual fuerte: prob. entró a edificio")
            reward += self.REWARDS["building_entry"]
            # Forzamos la salida
            for _ in range(10):
                move("down")
                time.sleep(1)
                exit_img = self.capture_and_process()
                if not self.image_changed(prev_img, exit_img, threshold=self.BUILDING_THRESHOLD):
                    # La puerta estará justo arriba
                    door_coord = self._future_coord('up')
                    self.world_map.update_tile(
                        door_coord, TileType.DOOR)
                    if self.save_mode:
                        self.world_map.save()
                    reward += self.REWARDS["building_exit"]
                    if debug:
                        print(
                            f"[DEBUG] Salió del edificio. Puerta registrada. Recompensa: {self.REWARDS['building_exit']}")
                    break
            else:
                reward += self.REWARDS["oak_zone_penalty"]
                if debug:
                    print(
                        f"[DEBUG] No salió del edificio. Penalización: {self.REWARDS['oak_zone_penalty']}")

        # 8) Nuevo estado
        state = self.get_state()
        return state, reward, False
