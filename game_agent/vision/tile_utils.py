# game_agent/vision/tile_utils.py
import cv2
import numpy as np
from config import OBSTACLE_THRESHOLD, PLAYER_POSITION, TILE_HEIGHT, TILE_WIDTH, WHITE_THRESHOLD

# Desplazamientos globales: ajusta estos valores hasta que la cuadrícula
# roja cuadre perfectamente con tu personaje en pantalla.
X_OFFSET = 0
Y_OFFSET = 0


def split_into_tiles(image,
                     tile_height: int = TILE_HEIGHT,
                     tile_width:  int = TILE_WIDTH):
    """
    Divide `image` en tiles de tamaño (tile_height x tile_width),
    empezando en (X_OFFSET, Y_OFFSET).
    Retorna una lista de filas, cada fila es lista de np.ndarray.
    """
    h, w = image.shape[:2]
    tiles = []
    # Barrido vertical, partiendo de Y_OFFSET
    for top in range(Y_OFFSET, h, tile_height):
        row = []
        # Barrido horizontal, partiendo de X_OFFSET
        for left in range(X_OFFSET, w, tile_width):
            tile = image[top: top + tile_height,
                         left: left + tile_width]
            row.append(tile)
        tiles.append(row)
    return tiles


def is_tile_free(tile: np.ndarray, white_threshold: float = 0.2, debug: bool = True) -> bool:
    """
    Decide si un único tile es “libre” (blanco / transitable) o “obstáculo” (oscuro)
    basándose en su brillo medio.

    Parámetros:
      - tile: parche de imagen (uint8 0–255 o float 0–1).
      - white_threshold: umbral de brillo medio (0–1) para considerarlo libre.

    Retorna:
      - True si mean(tile) > white_threshold → suelo/transitable.
      - False si mean(tile) <= white_threshold → pared/obstáculo.
    """
    # Convertimos siempre a float 0–1
    arr = tile.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0

    mean_val = arr.mean()
    if debug:
        print(f"[DEBUG] mean(tile) = {mean_val:.2f} "
              f"→ {'Libre' if mean_val > white_threshold else 'Obstáculo'}")
    return mean_val > white_threshold


def get_surrounding_obstacles(tiles, player_pos=PLAYER_POSITION, white_threshold: int = OBSTACLE_THRESHOLD, debug: bool = True):
    """
    Detecta si hay obstáculos en las direcciones cardinales desde la posición del jugador,
    comprobando sólo el tile adyacente en cada dirección, en el rango 0–255.

    Parámetros:
        tiles: matriz 2D de tiles (listas de listas de np.ndarray)
        player_pos: tupla (col, fila) del tile del jugador
        white_threshold: umbral [0–255] para considerar un tile “libre” (blanco)
        debug: si True, imprime información de los tiles evaluados

    Retorna:
        dict con claves 'up', 'down', 'left', 'right' y valores booleanos
        (True si hay obstáculo i.e. mean(tile) < white_threshold)
    """
    cx, cy = player_pos

    # offsets (dx, dy) relativos a la posición del jugador
    dir_map = {
        'up':    (0, -1),
        'down':  (0, +1),
        'left':  (-1,  0),
        'right': (+1,  0),
    }

    obstacles = {}
    for direction, (dx, dy) in dir_map.items():
        tx, ty = cx + dx, cy + dy

        # extraer el tile si está dentro del rango, o un parche negro si no
        if 0 <= ty < len(tiles) and 0 <= tx < len(tiles[0]):
            tile = tiles[ty][tx]
        else:
            tile = np.zeros_like(tiles[0][0])

        # calculamos el brillo medio directamente en [0–255]
        mean_val = float(np.mean(tile))
        # si es menor al umbral → obstáculo
        found_obstacle = mean_val < white_threshold

        obstacles[direction] = found_obstacle

        if debug:
            print(f"[DEBUG] Dirección '{direction}':")
            # print(f"   Posición tile = {(tx, ty)}")
            print(f"   → mean = {mean_val:.2f}")
            print(
                f"   → {'Obstáculo' if found_obstacle else 'Libre'} (umbral={white_threshold})\n")

    return obstacles


def overlay_red_grid(image,
                     tile_height: int = TILE_HEIGHT,
                     tile_width:  int = TILE_WIDTH):
    """
    Dibuja sobre `image` (grayscale o RGB) una cuadrícula roja con etiquetas,
    usando X_OFFSET/Y_OFFSET idénticos a split_into_tiles.
    """
    # Convertir a RGB si es grayscale
    if image.ndim == 2:
        image_rgb = np.stack([image]*3, axis=-1).astype(np.uint8)
    else:
        image_rgb = image.copy()

    h, w = image_rgb.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (0, 0, 255)  # BGR rojo

    # Líneas horizontales y etiquetas de fila
    y = Y_OFFSET
    row = 0
    while y < h:
        cv2.line(image_rgb, (X_OFFSET, y), (w, y), color, 1)
        cv2.putText(image_rgb,
                    str(row),
                    (X_OFFSET + 2, y + 12),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        y += tile_height
        row += 1

    # Líneas verticales y etiquetas de columna
    x = X_OFFSET
    col = 0
    while x < w:
        cv2.line(image_rgb, (x, Y_OFFSET), (x, h), color, 1)
        cv2.putText(image_rgb,
                    str(col),
                    (x + 2, Y_OFFSET + 12),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        x += tile_width
        col += 1

    return image_rgb
