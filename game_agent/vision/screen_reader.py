# screen_reader.py
import pyautogui
import numpy as np
import cv2
import os
from datetime import datetime


def capture_region(region=None):
    """
    Capture a screenshot of an specific region of the screen

    Args:
        region (tuple): (left, top, width, height). If None, goes full screen.
    """
    screenshot = pyautogui.screenshot(region=region)
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return image


def save_image(image, label="debug", folder="./game_agent/vision/debug_captures"):
    """
    Saves image with timestamp and label.

    Args:
        image (np.array): Image itself.
        label (str): Image label.
        folder (str): Folder route.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.png"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    print(f"[DEBUG] Imagen guardada: {path}")
