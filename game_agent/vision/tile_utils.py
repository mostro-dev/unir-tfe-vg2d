# game_agent/vision/tile_utils.py

import cv2
import matplotlib.pyplot as plt
import numpy as np


def split_into_tiles(image, tile_height, tile_width):
    rows = image.shape[0] // tile_height
    cols = image.shape[1] // tile_width
    tile_grid = []

    for r in range(rows):
        row = []
        for c in range(cols):
            tile = image[
                r * tile_height: (r + 1) * tile_height,
                c * tile_width: (c + 1) * tile_width
            ]
            row.append(tile)
        tile_grid.append(row)

    return tile_grid


def compare_tiles(tile1, tile2, threshold=0.2):
    if tile1.shape != tile2.shape:
        return True

    if tile1.max() > 1.0:
        tile1 = tile1 / 255.0
    if tile2.max() > 1.0:
        tile2 = tile2 / 255.0

    diff = np.mean(np.abs(tile1 - tile2))
    return diff > threshold


def get_surrounding_obstacles(tiles, player_top_left=(3, 4), white_threshold=180, debug=False):
    """
    Detecta si hay obstáculos en las direcciones cardinales desde el bloque 2x2 del jugador.

    Parámetros:
        tiles: matriz 2D de tiles (listas de listas de np.ndarray)
        player_top_left: tupla (fila, columna) del tile superior izquierdo del jugador
        white_threshold: umbral para considerar si un tile es blanco/transitable
        debug: si True, imprime información de los tiles evaluados

    Retorna:
        dict con claves 'up', 'down', 'left', 'right' y valores booleanos (True si hay obstáculo)
    """
    r, c = player_top_left

    directions = {
        'up':    [(r-1, c), (r-1, c+1)],
        'down':  [(r+2, c), (r+2, c+1)],
        'left':  [(r, c-1), (r+1, c-1)],
        'right': [(r, c+2), (r+1, c+2)],
    }

    obstacles = {}

    for direction, positions in directions.items():
        means = []
        for row, col in positions:
            if (
                row < 0 or row >= len(tiles) or
                col < 0 or col >= len(tiles[0])
            ):
                means.append(float('nan'))  # Marcamos fuera de rango como NaN
                continue
            tile = tiles[row][col]
            mean_val = np.mean(tile)
            means.append(mean_val)

        combined_mean = np.nanmean(means)
        found_obstacle = combined_mean < white_threshold

        obstacles[direction] = found_obstacle

        if debug:
            print(f"[DEBUG] Dirección '{direction}':")
            for i, (pos, val) in enumerate(zip(positions, means)):
                print(f"   Tile {i+1} en {pos} → mean = {val:.2f}")
            print(f"   → Promedio combinado = {combined_mean:.2f}")
            print(f"   → {'Obstáculo' if found_obstacle else 'Libre'}\n")

    return obstacles


def overlay_red_grid(image, tile_height=35, tile_width=32):
    """
    Dibuja una cuadrícula roja y añade etiquetas de fila y columna en rojo.

    Parámetros:
        image: np.ndarray (grayscale o RGB)
        tile_height: alto del tile
        tile_width: ancho del tile

    Retorna:
        imagen con cuadrícula y etiquetas
    """
    # Convertir a RGB si es una imagen en escala de grises
    if len(image.shape) == 2:
        image_rgb = np.stack([image]*3, axis=-1).astype(np.uint8)
    else:
        image_rgb = image.copy()

    height, width = image_rgb.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (0, 0, 255)  # 🔴 Rojo en formato BGR

    # Dibujar líneas horizontales y etiquetas de fila
    for y in range(0, height, tile_height):
        cv2.line(image_rgb, (0, y), (width, y), color, 1)
        row_idx = y // tile_height
        cv2.putText(image_rgb, str(row_idx), (2, y + 12), font,
                    font_scale, color, thickness, cv2.LINE_AA)

    # Dibujar líneas verticales y etiquetas de columna
    for x in range(0, width, tile_width):
        cv2.line(image_rgb, (x, 0), (x, height), color, 1)
        col_idx = x // tile_width
        cv2.putText(image_rgb, str(col_idx), (x + 2, 12), font,
                    font_scale, color, thickness, cv2.LINE_AA)

    return image_rgb
