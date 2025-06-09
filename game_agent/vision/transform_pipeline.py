from datetime import datetime
from skimage import exposure
from skimage import io, morphology, filters, feature, color
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

selem = morphology.rectangle(4, 4)
selem_2 = morphology.rectangle(8, 8)
folder = "./game_agent/vision/debug_captures_pipeline"
label = "debug"


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
    print(f"[DEBUG] Imagen guardada: {path}")
    # return result
