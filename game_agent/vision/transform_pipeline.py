from datetime import datetime
from skimage import exposure
from skimage import io, morphology, filters, feature, color
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from config import TILE_HEIGHT, TILE_WIDTH
from game_agent.vision.tile_utils import overlay_red_grid

selem = morphology.rectangle(4, 4)
selem_2 = morphology.rectangle(8, 8)
folder = "./game_agent/vision/debug_captures_pipeline"
folder_grid = "./game_agent/vision/debug_captures_grid"
label = "debug"
label_grid = "debug_grid"


def save_image_pipeline(image, agent_pos, debug: bool = True):
    low = 215
    high = 240
    mid = 100
    gray = color.rgb2gray(image)
    # 2) paso a uint8 [0,255]
    gray_u8 = (gray * 255).astype(np.uint8)

    # result = np.full_like(gray_u8, fill_value=mid, dtype=np.uint8)
    result = gray_u8
    result[gray_u8 < low] = 0
    # result[(gray_u8 >= low) & (gray_u8 < high)] = mid
    result[gray_u8 >= high] = 255

    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%H%M%S")
    x, y = agent_pos
    pos_label = f"{x:02d}_{y:02d}"
    filename = f"{label}_{timestamp}_{pos_label}.png"
    path = os.path.join(folder, filename)

    cv2.imwrite(path, result)

    # Agregar grid y guardar
    if not os.path.exists(folder_grid):
        os.makedirs(folder_grid)

    filename_grid = f"{label_grid}_{timestamp}.png"
    path_grid = os.path.join(folder_grid, filename_grid)
    result_with_grid = overlay_red_grid(gray_u8)
    cv2.imwrite(path_grid, result_with_grid)

    if debug:
        print(f"[DEBUG] Imagen guardada: {path} \n")
    return result
