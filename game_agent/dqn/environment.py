# game_agent/dqn/environment.py

from collections import deque
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
    IMAGE_CHANGED_THRESHOLD = 20  # umbral para detectar cambios visuales
    TILE_CHANGED_THRESHOLD = 8  # umbral para detectar cambios visuales
    OAK_STEPS_BACK = 2  # pasos hacia atrás en zona Oak

    REWARDS = {
        # Exploración:
        "move_success": +1.0,    # recompensa pequeña y constante
        "move_revisit": -0.2,    # penalización suave, pero no mata la exploración
        "move_wall": -1.0,    # aviso moderado al chocar
        "move_no_wall": 0.0,    # neutro si no se movió pero no era pared

        # Interacción / edificios:
        "interaction_success": +3.0,   # incentivo a interactuar
        "building_entry": +4.0,   # recompensa a descubrir edificio
        "building_exit": +1.0,   # extra al salir

        # Peligros y mal uso:
        "oak_zone_penalty": -2.0,   # penaliza entrar en zona Oak
        "spammed_a_button": -1.0,   # desalienta spam de Z
    }

    def __init__(self, save_mode: bool = True, punish_revisit: bool = True):
        # flag que indica si persistimos el world_map en disco
        self.save_mode = save_mode
        self.punish_revisit = punish_revisit

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

        # para detectar spam de Z
        self.last_action = None
        self.action_history = deque(maxlen=3)

        # Para registrar los últimos tiles visitados
        self.recent_tiles = deque(maxlen=8)
        self.recent_tiles.append(self.agent_pos)

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

    def handle_oak_zone(self, current_image, debug=True):
        """
        Si la zona de Oak está activa, fuerza a bajar self.OAK_STEPS_BACK tiles
        y penaliza, actualizando también la posición lógica y el mapa.
        """
        if is_oak_zone_triggered(current_image, debug):
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
            print(
                f"[DEBUG] Diferencia imágenes es en promedio: {mean_diff:.2f}")
        return mean_diff > threshold

    def near_tile_changed(self, direction, prev_img, new_img, debug=True):
        """
        Comprueba si el tile en `direction` ha cambiado desde la última vez.
        Devuelve True si ha cambiado, False si no.
        """
        if debug:
            print(f"[DEBUG] Comprobando cambio en tile {direction}")

        if direction in ['up', 'down', 'left', 'right']:
            tiles_prev = split_into_tiles(
                prev_img, self.tile_height, self.tile_width)
            tiles_new = split_into_tiles(
                new_img,  self.tile_height, self.tile_width)
            dir_map = {
                "up":    [(-1, -2), (0, -2), (-1, -1), (0, -1)],
                "down":  [(-1, +2), (0, +2), (-1, +3), (0, +3)],
                "left":  [(-2, -1), (-2, 0),  (-1, -1), (-1, 0)],
                "right": [(+2, -1), (+2, 0),  (+3, -1), (+3, 0)],
            }
            cx, cy = self.agent_pos
            prev_patches, new_patches = [], []
            for dx, dy in dir_map[direction]:
                tx, ty = cx + dx, cy + dy
                if 0 <= ty < len(tiles_prev) and 0 <= tx < len(tiles_prev[0]):
                    prev_patches.append(tiles_prev[ty][tx])
                    new_patches.append(tiles_new[ty][tx])
                else:
                    zero = np.zeros_like(tiles_prev[0][0])
                    prev_patches.append(zero)
                    new_patches.append(zero)
            # stackea y diff medio
            prev_stack = np.stack(prev_patches).astype(np.int16)
            new_stack = np.stack(new_patches).astype(np.int16)
            diff_patch = np.mean(np.abs(prev_stack - new_stack))
            patch_moved = diff_patch > self.TILE_CHANGED_THRESHOLD
            if debug:
                print(
                    f"[DEBUG] diff patch = {diff_patch:.2f} → patch_moved={patch_moved}")

        return patch_moved if direction in ['up', 'down', 'left', 'right'] else False

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

    def is_real_obstacle(self, direction, old_image, threshold=12, debug=True, dialog_debug=True):
        """
        Comprueba si tras intentar moverse e interactuar sigue bloqueado:
        si es pared → devuelve True (guarda WALL),
        si no → False (guarda INFO).
        """
        if debug:
            print(f"[DEBUG] Comprobando obstáculo real en {direction}")
        before = old_image
        time.sleep(1)  # espera para estabilizar captura
        after = self.capture_and_process()

        future_coord = self._future_coord(direction)

        global_image_changed = self.image_changed(
            before, after, self.IMAGE_CHANGED_THRESHOLD, debug)

        local_image_changed = self.near_tile_changed(
            direction, before, after, debug)

        # Si se movió, no era obstáculo
        if global_image_changed and local_image_changed:
            if debug:
                print(f"[DEBUG] Movió con éxito a {direction}")
            return False

        # Intentar interactuar
        press('z')
        time.sleep(2)

        is_dialog_open = is_dialog_open_by_template(
            capture_region(GAME_REGION),
            debug=dialog_debug
        )
        had_dialog_open = is_dialog_open
        if is_dialog_open:
            self.world_map.update_tile(future_coord, TileType.INFO)
        if debug:
            print("[DEBUG] Entrando en bucle de texto")

        # bucle hasta que el diálogo se cierre
        while True:
            if not is_dialog_open:
                break
            press('z')
            time.sleep(0.5)
            is_dialog_open = is_dialog_open_by_template(
                capture_region(GAME_REGION),
                debug=dialog_debug
            )

        # si al principio hubo diálogo, es INFO (return False), si no, es WALL (return True)
        if had_dialog_open:
            if debug:
                print(
                    f"[DEBUG] Confirma INFO en {direction}, agent_pos={self.agent_pos}"
                )
            return False
        else:
            if debug:
                print(
                    f"[DEBUG] Confirma WALL en {direction}, agent_pos={self.agent_pos}"
                )
            return True

    def get_state(self):
        # 1) los 4 obstáculos originales
        proc = self.capture_and_process()
        tiles = split_into_tiles(proc, self.tile_height, self.tile_width)
        obs = get_surrounding_obstacles(tiles, player_top_left=(4, 3))
        basic = np.array([float(obs[d]) for d in ("up", "right", "down", "left")],
                         dtype=np.float32)

        # 2) ahora extraigo del mapa local las probabilidades para cada vecino
        extras = []
        for direction in ("up", "right", "down", "left"):
            coord = self._future_coord(direction)
            entry = self.world_map.map.get(coord, {})
            # ejemplo: FLOOR, INFO, DOOR, WALL, _visits
            extras.append(entry.get(TileType.FLOOR.value, 0.0))
            extras.append(entry.get(TileType.INFO.value,  0.0))
            extras.append(entry.get(TileType.DOOR.value,  0.0))
            extras.append(entry.get(TileType.WALL.value,  0.0))
            # podrías normalizar _visits si te interesa:
            extras.append(entry.get("_visits", 0) / 10.0)

        return np.concatenate([basic, np.array(extras, dtype=np.float32)])

    def step(self, action: str, debug: bool = True):
        """
        Ejecuta la acción, calcula recompensa, actualiza posición lógica
        AL FINAL, y registra el tile en world_map (con penalización por revisitas).
        Detección de movimiento combinando diff global y diff local de parches.
        """
        # --- 0) Debug inicial ---
        if debug:
            print(f"\n[DEBUG] Acción a ejecutar: {action}")
            time.sleep(1)
            print(f"[DEBUG] Antes de step – agent_pos={self.agent_pos}")

        # --- 1) Captura previa y Oak-zone ---
        prev_img = self.capture_and_process()
        triggered, oak_penalty = self.handle_oak_zone(prev_img, debug=debug)
        if triggered:
            if debug:
                print(
                    f"[DEBUG] Tras Oak Zone – agent_pos={self.agent_pos}, penalidad={oak_penalty}")
            return self.get_state(), oak_penalty, False

        # --- 2) Coordenada candidata sin actualizar aún ---
        next_coord = self._future_coord(action)
        if debug:
            print(
                f"[DEBUG] Acción candidata: {action}, next_coord={next_coord}")

        time.sleep(1)

        # --- 3) Ejecutamos la acción física ---
        if action in ['up', 'down', 'left', 'right']:
            move(action)
            self.last_direction = action
        else:  # 'z'
            press('z')
            time.sleep(1)
        print(
            f"\n[DEBUG] Acción ejecutada: {action} desde la casilla={self.agent_pos}")

        # --- 4) Captura tras la acción ---
        new_img = self.capture_and_process()

        # --- 5.1) Diff global para detectar movimiento general ---
        global_moved = self.image_changed(
            prev_img, new_img, threshold=self.IMAGE_CHANGED_THRESHOLD, debug=debug)
        if debug:
            print(f"[DEBUG] diff global → moved_global={global_moved}")

        moved = global_moved

        # --- 5.2) Gate: si next_coord ya es muro, fuerza no movimiento ---
        entry = self.world_map.map.get(next_coord, {})
        # Accedemos al count de 'WALL' dentro de '_counts'
        wall_count = entry.get('_counts', {}).get('WALL', 0)

        # Obtenemos el valor de 'WALL' directamente del diccionario principal
        wall_value = entry.get('WALL', 0.0)

        if wall_count >= 2 and wall_value >= 1:
            # if entry.get(TileType.WALL.value, 0.0) >= 1.0:
            moved = False
            if debug:
                print(
                    f"[DEBUG] next_coord {next_coord} marcado como WALL → moved=False")

        # --- 6) Spam de Z ---
        reward = 0.0
        self.action_history.append(action)
        if self.action_history.count('z') > 2:
            reward += self.REWARDS["spammed_a_button"]
            if debug:
                print(
                    f"[DEBUG] Spam de Z detectado. Penalización: {self.REWARDS['spammed_a_button']}")

        # --- 7) Movimiento: asignamos recompensa y actualizamos posición al final ---
        if action in ['up', 'down', 'left', 'right']:
            if moved:
                x, y = next_coord
                # clamp para no-negativos
                if x < 0 or y < 0:
                    moved = False
                    if debug:
                        print(
                            f"[DEBUG] next_coord {next_coord} fuera de límites → moved=False")
                    # raise ValueError(
                    #     f"next_coord {next_coord} fuera de límites: x={x}, y={y}")
                else:
                    # si se movió, chequeamos si es revisita
                    if self.recent_tiles.__contains__(next_coord) and self.punish_revisit:
                        reward += self.REWARDS["move_revisit"]
                        if debug:
                            print(
                                f"[DEBUG] Revisitando {next_coord}. Penalización: {self.REWARDS['move_revisit']}"
                            )
                    else:
                        reward += self.REWARDS["move_success"]
                        if debug:
                            print(
                                f"[DEBUG] Movido con éxito a {next_coord}. Recompensa: {self.REWARDS['move_success']}"
                            )
                        # marcamos provisionalmente como FLOOR
                        self.world_map.update_tile(next_coord, TileType.FLOOR)
                    # finalmente actualizamos posición lógica
                    self.agent_pos = next_coord
                    if debug:
                        print(
                            f"\n[DEBUG] agent_pos actualizado a {self.agent_pos}")
                    self.world_map.mark_visited(self.agent_pos)
                    if self.save_mode:
                        self.world_map.save()
                self.recent_tiles.append(self.agent_pos)
            else:
                # sin movimiento: chequeo obstáculo real
                if self.is_real_obstacle(action, old_image=prev_img, debug=debug):
                    reward += self.REWARDS["move_wall"]
                    self.world_map.update_tile(next_coord, TileType.WALL)
                    if self.save_mode:
                        self.world_map.save()
                    if debug:
                        print(
                            f"[DEBUG] Choque pared con {action}. Recompensa: {self.REWARDS['move_wall']}"
                        )

        # --- 8) Interacción con 'z' (sin cambios) ---
        else:
            frame = capture_region(GAME_REGION)
            dialog = is_dialog_open_by_template(frame)
            if dialog and not self.is_text_in_screen:
                reward += self.REWARDS["interaction_success"]
                self.is_text_in_screen = True
                if debug:
                    print(
                        f"[DEBUG] Interacción exitosa con Z. Recompensa: {self.REWARDS['interaction_success']}"
                    )
                coord = self._future_coord(self.last_direction)
                self.world_map.update_tile(coord, TileType.INFO)
                if debug:
                    print(
                        f"[DEBUG] Guardando INFO en {coord} (agent_pos={self.agent_pos}"
                    )
                if self.save_mode:
                    self.world_map.save()
                # Mantener pulsando Z hasta cerrar diálogo
                while True:
                    press('z')
                    time.sleep(1)
                    if not is_dialog_open_by_template(capture_region(GAME_REGION)):
                        if debug:
                            print("[DEBUG] Diálogo cerrado.")
                        self.is_text_in_screen = False
                        break
            elif not dialog and self.is_text_in_screen:
                self.is_text_in_screen = False

        # --- 9) Detección de edificio (igual que antes) ---
        if self.image_changed(prev_img, new_img, threshold=self.BUILDING_THRESHOLD, debug=debug):
            if debug:
                print("[DEBUG] Cambio visual fuerte: posible edificio")
            reward += self.REWARDS["building_entry"]
            for _ in range(10):
                move("down")
                time.sleep(1)
                exit_img = self.capture_and_process()
                if not self.image_changed(prev_img, exit_img, threshold=self.BUILDING_THRESHOLD, debug=debug):
                    door_coord = self._future_coord('up')
                    self.world_map.update_tile(door_coord, TileType.DOOR)
                    reward += self.REWARDS["building_exit"]
                    if debug:
                        print(
                            f"[DEBUG] Salió del edificio. Recompensa: {self.REWARDS['building_exit']} - agent_pos={self.agent_pos}"
                        )
                    if self.save_mode:
                        self.world_map.save()
                    break
            else:
                reward += self.REWARDS["oak_zone_penalty"]
                if debug:
                    print(
                        f"[DEBUG] No salió del edificio. Penalización: {self.REWARDS['oak_zone_penalty']} - agent_pos={self.agent_pos}"
                    )

        time.sleep(3)

        # --- 10) Retorno ---
        if debug:
            print(f"\n[DEBUG] Posición final del agente: {self.agent_pos}")
            print(f"[DEBUG] Paso completo – reward acumulada: {reward:.2f}\n")
            print(f"\n\n////// -- Fin Step -- //////\n\n")
        return self.get_state(), reward, False
