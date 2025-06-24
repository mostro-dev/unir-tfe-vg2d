# game_agent/dqn/environment.py

from collections import deque
import glob
import os
import time
import cv2
import numpy as np
from datetime import datetime

import openai

from config import DIRECTIONS, DIRECTIONS_TUPLE, GAME_REGION, OBSTACLE_THRESHOLD, PLAYER_POSITION, TILE_HEIGHT, TILE_WIDTH, TileType
from game_agent.map.world_map import WorldMap
from game_agent.controller.keyboard_controller import move, press
from game_agent.vision.screen_reader import capture_region, save_image
from game_agent.vision.transform_pipeline import save_image_pipeline
from game_agent.vision.tile_utils import split_into_tiles, get_surrounding_obstacles
from game_agent.vision.dialog_detector import is_dialog_open_by_template
from game_agent.vision.oak_zone_detector import is_oak_zone_triggered


class GameEnvironment:
    BUILDING_THRESHOLD = 85  # umbral para detectar entrada a edificio
    IMAGE_CHANGED_THRESHOLD = 17  # umbral para detectar cambios visuales
    TILE_CHANGED_THRESHOLD = 8  # umbral para detectar cambios visuales
    OAK_STEPS_BACK = 2  # pasos hacia atrás en zona Oak
    OAK_THRESHOLD = 0.8
    DOOR_STEPS_BACK = 2  # pasos hacia atrás al salir de un edificio
    # umbral para considerar un tile como obstáculo
    TILE_OBSTACLE_THRESHOLD = OBSTACLE_THRESHOLD

    REWARDS = {
        # Exploración:
        "move_success": +1.0,    # recompensa pequeña y constante
        "move_revisit": -0.2,    # penalización suave, pero no mata la exploración
        "move_wall": -1.0,    # aviso moderado al chocar
        "move_no_wall": 0.0,    # neutro si no se movió pero no era pared

        # Interacción / edificios:
        "interaction_success": +0.5,   # incentivo a interactuar
        "building_entry": +0.3,   # recompensa a descubrir edificio
        "building_exit": +0.1,   # extra al salir

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

        self.last_direction = None  # última dirección de movimiento

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
        save_image(frame, self.agent_pos)
        return save_image_pipeline(frame, self.agent_pos)

    def handle_oak_zone(self, current_image, debug: bool = True):
        """
        Si la zona de Oak está activa, fuerza a bajar self.OAK_STEPS_BACK tiles
        y penaliza, actualizando también la posición lógica y el mapa.
        """
        if is_oak_zone_triggered(current_image, self.OAK_THRESHOLD, debug):
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

    def image_changed(self, img1, img2, threshold=IMAGE_CHANGED_THRESHOLD, debug: bool = True):
        """Compara diferencia media absoluta entre imágenes."""
        if img1.shape != img2.shape:
            return True
        diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
        mean_diff = np.mean(diff)
        if debug:
            print(
                f"[DEBUG] Diferencia imágenes es en promedio: {mean_diff:.2f}")
        return mean_diff > threshold

    def near_tile_changed(self, direction, prev_img, new_img, debug: bool = True):
        """
        Comprueba si el tile en `direction` ha cambiado desde la última vez.
        Devuelve True si ha cambiado, False si no.
        """
        if debug:
            print(f"[DEBUG] Comprobando cambio en tile {direction}")

        if direction in DIRECTIONS:
            # 1) Particionamos cada imagen en tiles
            tiles_prev = split_into_tiles(prev_img,
                                          self.tile_height,
                                          self.tile_width)
            tiles_new = split_into_tiles(new_img,
                                         self.tile_height,
                                         self.tile_width)

            # 2) Offset de un solo tile en la dirección deseada
            dir_map = {
                "up":    [(0, -1)],
                "down":  [(0, +1)],
                "left":  [(-1,  0)],
                "right": [(+1,  0)],
            }

            cx, cy = self.agent_pos
            prev_patches = []
            new_patches = []

            # 3) Extraemos ese único parche (o un parche negro si está fuera de rango)
            for dx, dy in dir_map[direction]:
                tx, ty = cx + dx, cy + dy
                if 0 <= ty < len(tiles_prev) and 0 <= tx < len(tiles_prev[0]):
                    prev_patches.append(tiles_prev[ty][tx])
                    new_patches.append(tiles_new[ty][tx])
                else:
                    zero = np.zeros_like(tiles_prev[0][0])
                    prev_patches.append(zero)
                    new_patches.append(zero)

            # 4) Apilamos y calculamos la diferencia media absoluta
            prev_stack = np.stack(prev_patches).astype(np.int16)
            new_stack = np.stack(new_patches).astype(np.int16)
            diff_patch = np.mean(np.abs(prev_stack - new_stack))

            # 5) Comparamos con el umbral
            patch_moved = diff_patch > self.TILE_CHANGED_THRESHOLD
            if debug:
                print(f"[DEBUG] diff patch = {diff_patch:.2f} "
                      f"→ patch_moved={patch_moved}")

        # Si la dirección no es válida, devolvemos False
        return patch_moved

    def extract_tile_in_direction(self, direction: str) -> np.ndarray:
        """
        Extrae el tile adyacente en la dirección indicada, relativa
        a la posición FÍSICA del jugador en pantalla (PLAYER_POSITION),
        no a su coord lógica en el mundo.
        Si está fuera de pantalla, devuelve un parche de ceros.
        """
        # 1) Captura con pipeline
        frame = self.capture_and_process()

        # 2) Divídelo en tiles
        tiles = split_into_tiles(frame, self.tile_height, self.tile_width)

        # 3) Posición fija del jugador en la retícula de pantalla
        px, py = PLAYER_POSITION  # (col, row) sobre la matriz tiles[row][col]

        # 4) Offset de un tile en la dirección deseada
        dir_map = {
            "up":    (0, -1),
            "down":  (0, +1),
            "left":  (-1,  0),
            "right": (+1,  0),
        }
        dx, dy = dir_map.get(direction, (0, 0))
        sx, sy = px + dx, py + dy

        # 5) Si está dentro de la rejilla, lo devolvemos; si no, zeros
        if 0 <= sy < len(tiles) and 0 <= sx < len(tiles[0]):
            return tiles[sy][sx]
        else:
            # parche negro del mismo tamaño
            h, w = self.tile_height, self.tile_width
            chan = frame.shape[2] if frame.ndim == 3 else 1
            shape = (h, w, chan) if chan > 1 else (h, w)
            return np.zeros(shape, dtype=frame.dtype)

    def save_tile_sample(self, direction: str, label: str, debug: bool = True):
        """
        Guarda en disco el tile CRUDO adyacente al sprite (usando PLAYER_POSITION),
        con la etiqueta `label` ("WALL", "INFO", "DOOR", etc.).
        """
        # Extrae el tile RAW
        tile = self.extract_tile_in_direction(direction)

        # Carpeta y nombre
        folder = os.path.join("tiles", label.upper())
        os.makedirs(folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{direction}_{ts}.png"
        path = os.path.join(folder, filename)

        # Asegura uint8 y guarda
        if tile.dtype != np.uint8:
            tile = tile.astype(np.uint8)
        cv2.imwrite(path, tile)

        if debug:
            mn, mx = int(tile.min()), int(tile.max())
            print(f"[DEBUG] Sample RAW '{label}' '{direction}' → {path}")
            print(
                f"        shape={tile.shape}, dtype={tile.dtype}, min={mn}, max={mx}")

    def has_obstacle_in_direction(self,
                                  direction: str,

                                  debug: bool = True) -> bool:
        """
        Mira el tile adyacente  y cuenta qué proporción de píxeles
        están por debajo de 'white_threshold'.
        Si esa proporción > ratio_threshold devuelve True (obstáculo).
        """
        tile = self.extract_tile_in_direction(direction)

        # contamos píxeles «oscuros»
        mean_val = float(np.mean(tile))
        found_obstacle = mean_val < self.TILE_OBSTACLE_THRESHOLD

        if debug:
            print(
                f"[DEBUG] Tile en '{direction}' es {'Obstáculo' if found_obstacle else 'Libre'} ")
            print(
                f"   → mean = {mean_val:.2f} (umbral={self.TILE_OBSTACLE_THRESHOLD})")

        return found_obstacle

    def is_real_obstacle(self, direction, old_image, threshold=IMAGE_CHANGED_THRESHOLD, debug: bool = True, dialog_debug: bool = True):
        """
        Comprueba si tras intentar moverse e interactuar sigue bloqueado:
        si es pared → devuelve True (guarda WALL),
        si no → False (guarda INFO).
        """

        # --- 0) Debug inicial ---
        if debug:
            print(f"[DEBUG] Comprobando obstáculo real en {direction}")

        # --- 1) Mapa local ---
        next_coord = self._future_coord(direction)
        entry = self.world_map.map.get(next_coord, {})
        floor_confidence = entry.get(TileType.FLOOR.value, 0.0)
        floor_count = entry.get('_counts', {}).get('WALL', 0)
        if floor_confidence >= 7.0 and floor_count < 5:
            if debug:
                print(
                    f"[DEBUG] world_map ya sabe que {next_coord} es FLOOR → no obstáculo")
            return "FLOOR"

        before = old_image
        time.sleep(1)  # espera para estabilizar captura
        after = self.capture_and_process()

        future_coord = self._future_coord(direction)

        global_image_changed = self.image_changed(
            before, after, threshold, debug)

        local_image_changed = self.near_tile_changed(
            direction, before, after, debug)

        # Si se movió, no era obstáculo
        if global_image_changed and local_image_changed:
            if debug:
                print(f"[DEBUG] Movió con éxito a {direction}")
            return "FLOOR"

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
        else:
            if not self.has_obstacle_in_direction(direction, debug=debug):
                if debug:
                    print(
                        f"[DEBUG] Tile adyacente en {direction} indica FLOOR")
                return "FLOOR"

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
            return "INFO"
        else:
            if debug:
                print(
                    f"[DEBUG] Confirma WALL en {direction}, agent_pos={self.agent_pos}"
                )
            return "WALL"

    def get_state(self):
        """
        Estado = [4 obstáculos inmediatos] +
                 [5 entradas por vecino * 4 vecinos] +
                 [ventana local H×W aplanada con 5 canales]
        donde H = número de filas de tiles visibles (9),
              W = número de columnas de tiles visibles (10).
        """
        # 1) Captura y partición en tiles procesados
        proc = self.capture_and_process()
        tiles = split_into_tiles(proc, self.tile_height, self.tile_width)

        # 2) Obstáculos inmediatos
        obs = get_surrounding_obstacles(tiles, player_pos=PLAYER_POSITION)
        basic = np.array([float(obs[d]) for d in DIRECTIONS_TUPLE],
                         dtype=np.float32)

        # 3) Características de los 4 vecinos
        extras = []
        for d in DIRECTIONS_TUPLE:
            coord = self._future_coord(d)
            entry = self.world_map.map.get(coord, {})
            extras.extend([
                entry.get(TileType.FLOOR.value, 0.0),
                entry.get(TileType.INFO.value,  0.0),
                entry.get(TileType.DOOR.value,  0.0),
                entry.get(TileType.WALL.value,  0.0),
                entry.get("_visits", 0) / 10.0,
            ])

        # 4) Dimensiones dinámicas de la ventana
        H = len(tiles)       # p.ej. 9 filas
        W = len(tiles[0])    # p.ej. 10 columnas
        half_h = H // 2
        half_w = W // 2

        # Construimos desplazamientos para cubrir exactamente W×H
        dys = [i - half_h for i in range(H)]
        dxs = [j - half_w for j in range(W)]

        # 5) Creamos el tensor ventana (H, W, 5)
        window = np.zeros((H, W, 5), dtype=np.float32)
        cx, cy = self.agent_pos
        for i, dy in enumerate(dys):
            for j, dx in enumerate(dxs):
                x, y = cx + dx, cy + dy
                entry = self.world_map.map.get((x, y), {})
                window[i, j, 0] = entry.get(TileType.FLOOR.value, 0.0)
                window[i, j, 1] = entry.get(TileType.INFO.value,  0.0)
                window[i, j, 2] = entry.get(TileType.DOOR.value,  0.0)
                window[i, j, 3] = entry.get(TileType.WALL.value,  0.0)
                window[i, j, 4] = entry.get("_visits", 0) / 10.0

        flat_win = window.ravel()

        # 6) Concatenamos todo en el vector de estado
        return np.concatenate([
            basic,                        # 4
            np.array(extras, dtype=np.float32),  # 20
            flat_win                      # H*W*5 = 9*10*5 = 450
        ])

    def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """
        Busca un camino por BFS desde start hasta goal sobre world_map,
        evitando casillas mapeadas como WALL.
        Devuelve la lista de coordenadas (incluyendo start y goal), o None si no hay ruta.
        """
        if start == goal:
            return [start]

        # Vecinos en orden (para BFS uniforme)
        directions = {'up': (0, -1), 'right': (1, 0),
                      'down': (0, 1), 'left': (-1, 0)}
        visited = set([start])
        queue = deque([[start]])

        while queue:
            path = queue.popleft()
            x, y = path[-1]
            for dx, dy in directions.values():
                nx, ny = x+dx, y+dy
                coord = (nx, ny)
                if coord in visited:
                    continue
                entry = self.world_map.map.get(coord, {})
                # Saltamos si sabemos que es pared
                if entry.get(TileType.WALL.value, 0) >= 1:
                    continue
                visited.add(coord)
                new_path = path + [coord]
                if coord == goal:
                    return new_path
                queue.append(new_path)
        return None

    def path_to_actions(self, path: list[tuple[int, int]]) -> list[str]:
        """
        Convierte una ruta de coordenadas [(x0,y0),(x1,y1),…] en
        una lista de acciones ['up','right',…].
        """
        actions = []
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            dx, dy = x1-x0, y1-y0
            if dx == 1 and dy == 0:
                actions.append('right')
            elif dx == -1 and dy == 0:
                actions.append('left')
            elif dx == 0 and dy == 1:
                actions.append('down')
            elif dx == 0 and dy == -1:
                actions.append('up')
            else:
                raise ValueError(
                    f"Movimiento no cardinal: {(x0,y0)} → {(x1,y1)}")
        return actions

    def step(self, action: str | tuple[int, int], debug: bool = True):
        """
        Ejecuta la acción (o planifica si le pasas una tupla destino), calcula recompensa,
        actualiza posición lógica AL FINAL, y registra el tile en world_map (con penalización por revisitas).
        Detección de movimiento combinando diff global y diff local de parches.
        """
        # --- Planificación de objetivo alto ---
        if isinstance(action, tuple):
            goal = action
            if debug:
                print(f"\n[DEBUG] Orden de planificación: ir a {goal}")
            path = self.find_path(self.agent_pos, goal)
            if path is None:
                if debug:
                    print(
                        f"[DEBUG] No se encontró ruta a {goal}, quedo en {self.agent_pos}")
                return self.get_state(), 0.0, False
            action_seq = self.path_to_actions(path)
            total_reward = 0.0
            for a in action_seq:
                # aquí recursivamente llamamos a step con strings
                _, r, _ = self.step(a, debug)
                total_reward += r
            if debug:
                print(
                    f"[DEBUG] Plan completado, recompensa total acumulada: {total_reward:.2f}")
            return self.get_state(), total_reward, False

        # Si llegamos aquí, action es un string atómico
        was_inside_building = False
        # --- 0) Debug inicial ---
        if debug:
            print(f"\n[DEBUG] Acción a ejecutar: {action}")
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

        time.sleep(0.5)

        # --- 3) Ejecutamos la acción física ---
        if action in ['up', 'down', 'left', 'right']:
            move(action)
            self.last_direction = action
        else:  # 'z'
            press('z')

        time.sleep(1.5)
        if debug:
            print(
                f"[DEBUG] Acción ejecutada: {action} desde la casilla={self.agent_pos}")

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
        wall_count = entry.get('_counts', {}).get('WALL', 0)
        wall_value = entry.get('WALL', 0.0)
        if wall_count >= 5 and wall_value >= 1:
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

        building_check = self.image_changed(
            prev_img, new_img, threshold=self.BUILDING_THRESHOLD, debug=debug)
        time.sleep(0.5)

        # --- 7) Movimiento: asignamos recompensa y actualizamos posición al final ---
        if action in DIRECTIONS:
            if moved:
                x, y = next_coord
                if x < -50 or y < -50:
                    # moved = False
                    if debug:
                        print(
                            f"[DEBUG] next_coord {next_coord} fuera de límites → moved=False")
                else:
                    if building_check:
                        if debug:
                            print(
                                "[DEBUG] Cambio visual fuerte: posible edificio")
                        reward += self.REWARDS["building_entry"]
                        was_inside_building = True
                        for _ in range(self.DOOR_STEPS_BACK):
                            move("down")
                            time.sleep(1)
                            exit_img = self.capture_and_process()
                            if not self.image_changed(prev_img, exit_img, threshold=self.BUILDING_THRESHOLD, debug=debug):
                                door_coord = self._future_coord('up')
                                self.world_map.update_tile(
                                    door_coord, TileType.DOOR)
                                self.save_tile_sample(
                                    direction='up', label="DOOR", debug=debug)
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
                        self.world_map.update_tile(next_coord, TileType.FLOOR)
                    if not was_inside_building:
                        self.agent_pos = next_coord

                        if debug:
                            print(
                                f"[DEBUG] agent_pos actualizado a {self.agent_pos}")
                        self.world_map.mark_visited(self.agent_pos)
                        if self.save_mode:
                            self.world_map.save()
                was_inside_building = False
                self.recent_tiles.append(self.agent_pos)
            else:
                # sin movimiento: chequeo obstáculo real
                real_obstacle = self.is_real_obstacle(
                    action, old_image=prev_img, debug=debug)
                if debug:
                    print(f"[DEBUG] is_real_obstacle → {real_obstacle}")
                match real_obstacle:
                    case "FLOOR":
                        reward += self.REWARDS["move_success"]
                        self.agent_pos = next_coord
                        self.recent_tiles.append(self.agent_pos)
                        self.world_map.update_tile(next_coord, TileType.FLOOR)
                        if self.save_mode:
                            self.world_map.save()
                        if debug:
                            print(
                                f"[DEBUG] Sin obstaculo en {action}. Valida movimiento. Recompensa: {self.REWARDS['move_wall']}")
                    case "INFO":
                        self.world_map.update_tile(next_coord, TileType.INFO)
                        self.save_tile_sample(
                            direction=action, label="INFO", debug=debug)
                    case "WALL":
                        reward += self.REWARDS["move_wall"]
                        self.world_map.update_tile(next_coord, TileType.WALL)
                        self.save_tile_sample(
                            direction=action, label="WALL", debug=debug)
                        if self.save_mode:
                            self.world_map.save()
                        if debug:
                            print(
                                f"[DEBUG] Choque pared con {action}. Recompensa: {self.REWARDS['move_wall']}")

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
                if DIRECTIONS.__contains__(self.last_direction):
                    coord = self._future_coord(self.last_direction)
                    self.world_map.update_tile(coord, TileType.INFO)
                    self.save_tile_sample(
                        direction=self.last_direction, label="INFO", debug=debug)
                if debug:
                    print(
                        f"[DEBUG] Guardando INFO en {coord} (agent_pos={self.agent_pos}"
                    )
                if self.save_mode:
                    self.world_map.save()
                while True:
                    press('z')
                    time.sleep(0.5)
                    if not is_dialog_open_by_template(capture_region(GAME_REGION)):
                        if debug:
                            print("[DEBUG] Diálogo cerrado.")
                        self.is_text_in_screen = False
                        break
            elif not dialog and self.is_text_in_screen:
                self.is_text_in_screen = False

        # --- 10) Retorno ---
        if debug:
            print(f"\n[DEBUG] Posición final del agente: {self.agent_pos}")
            print(f"[DEBUG] Paso completo – reward acumulada: {reward:.2f}\n")
            print(f"\n\n////// -- Fin Step -- //////\n\n")
        return self.get_state(), reward, False

# NL Methods
    # def _parse_nl(self, instruction: str) -> tuple[int, int]:
    #     """
    #     Llama al LLM para extraer la coordenada objetivo (x,y)
    #     o bien una etiqueta (“casa”, “árbol”, etc.).
    #     Devuelve la casilla de destino en coords lógicas.
    #     """
    #     openai.api_key = OPENAI_API_KEY
    #     prompt = f"""
    #     El agente dispone de un mapa cuadriculado donde cada tile tiene coordenadas (x,y) con origen en (0,0).
    #     Le pido que convierta esta orden en un objetivo (x,y):
    #     "{instruction}"
    #     Devuélveme solo dos enteros separados por coma, p.ej. "4,7".
    #     Si no entiende o no hay objetivo, devuelve "UNKNOWN".
    #     """
    #     resp = openai.ChatCompletion.create(
    #         model="gpt-4o-mini",
    #         messages=[{"role": "system", "content": "You are a coordinate parser."},
    #                   {"role": "user", "content": prompt}],
    #         temperature=0.0,
    #     )
    #     text = resp.choices[0].message.content.strip()
    #     try:
    #         x_str, y_str = text.split(",")
    #         return int(x_str), int(y_str)
    #     except:
    #         raise ValueError(f"No pude parsear coordenadas de: {text!r}")

    # def execute_nl(self, instruction: str, debug: bool = True):
    #     """
    #     Toma un comando en NL, lo parsea a un destino, planea ruta y lo ejecuta.
    #     """
    #     if debug:
    #         print(f"[DEBUG] Interpretando instrucción: {instruction!r}")
    #     # 1) parsear objetivo
    #     try:
    #         goal = self._parse_nl(instruction)
    #     except ValueError as e:
    #         if debug:
    #             print(f"[DEBUG] Parser falló: {e}")
    #         return

    #     if debug:
    #         print(f"[DEBUG] Objetivo parseado: {goal}")

    #     # 2) buscar ruta en el mapa conocido
    #     path = self.find_path(self.agent_pos, goal)
    #     if path is None:
    #         if debug:
    #             print("[DEBUG] No hay ruta conocida hasta", goal)
    #         return

    #     actions = self.path_to_actions(path)
    #     if debug:
    #         print(f"[DEBUG] Ruta encontrada ({len(path)} pasos): {path}")
    #         print(f"[DEBUG] Acciones a ejecutar: {actions}")

    #     # 3) ejecutar cada paso con step()
    #     for a in actions:
    #         state, reward, done = self.step(a, debug=debug)
    #         if done:
    #             break

    #     if debug:
    #         print(
    #             f"[DEBUG] Llegamos a {self.agent_pos}, recompensa acumulada final: {reward:.2f}")
