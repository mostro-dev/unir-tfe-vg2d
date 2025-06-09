# game_agent/vision/tile_utils.py

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


def get_surrounding_obstacles(tiles, player_top_left=(3, 4), white_threshold=150, debug=False):
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
        found_obstacle = False
        for row, col in positions:
            if (
                row < 0 or row >= len(tiles) or
                col < 0 or col >= len(tiles[0])
            ):
                found_obstacle = True
                means.append(float('nan'))  # Marcamos fuera de rango como NaN
                continue
            tile = tiles[row][col]
            mean_val = np.mean(tile)
            means.append(mean_val)
            if mean_val < white_threshold:
                found_obstacle = True

        obstacles[direction] = found_obstacle

        if debug:
            print(f"[DEBUG] Dirección '{direction}':")
            for i, (pos, val) in enumerate(zip(positions, means)):
                print(f"   Tile {i+1} en {pos} → mean = {val:.2f}")
            combined_mean = np.nanmean(means)
            print(f"   → Promedio combinado = {combined_mean:.2f}")
            print(f"   → {'Obstáculo' if found_obstacle else 'Libre'}\n")

    return obstacles



def overlay_red_grid(image, tile_height=35, tile_width=32):
    """
    Dibuja una cuadrícula roja sobre una imagen dada.

    Args:
        image (np.ndarray): Imagen en escala de grises o RGB.
        tile_height (int): Altura de cada celda de la cuadrícula.
        tile_width (int): Ancho de cada celda de la cuadrícula.

    Returns:
        np.ndarray: Imagen con la cuadrícula roja superpuesta.
    """
    # Convertir a RGB si la imagen es en escala de grises
    if len(image.shape) == 2:
        image_rgb = np.stack([image]*3, axis=-1).astype(np.uint8)
    else:
        image_rgb = image.copy()

    height, width = image.shape[:2]

    # Dibujar líneas horizontales
    for y in range(0, height, tile_height):
        image_rgb[y:y+1, :] = [255, 0, 0]  # Línea horizontal roja

    # Dibujar líneas verticales
    for x in range(0, width, tile_width):
        image_rgb[:, x:x+1] = [255, 0, 0]  # Línea vertical roja

    return image_rgb
