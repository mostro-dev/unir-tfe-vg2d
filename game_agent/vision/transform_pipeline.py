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
folder = "./game_agent/vision/captures_pipeline"
label = "debug"
label_grid = "debug_grid"


def save_image_pipeline(image, low=0.7, high=0.85, mid=100):
    image = color.rgb2gray(image)
    dilated_image = morphology.dilation(image, selem)
    dilated_eroded_image = morphology.erosion(dilated_image, selem_2)

    result = np.full_like(dilated_eroded_image, fill_value=mid)
    result[dilated_eroded_image < low] = 0
    result[dilated_eroded_image > high] = 255

    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.png"
    path = os.path.join(folder, filename)

    cv2.imwrite(path, result)

    # Agregar grid y guardar
    filename_grid = f"{label_grid}_{timestamp}.png"
    path_grid = os.path.join(folder, filename_grid)
    result_with_grid = overlay_red_grid(result)
    cv2.imwrite(path_grid, result_with_grid)

    print(f"[DEBUG] Imagen guardada: {path}")
    return result
