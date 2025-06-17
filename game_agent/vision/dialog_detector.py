import cv2
import numpy as np
from game_agent.vision.screen_reader import capture_region
from config import GAME_REGION

# Carga la plantilla una vez
TEMPLATE_PATH = "./game_agent/vision/templates/text_box_template.png"
dialog_template = cv2.imread(TEMPLATE_PATH)
dialog_template_gray = cv2.cvtColor(dialog_template, cv2.COLOR_BGR2GRAY)


def extract_dialog_region(image, top_ratio=0.72, bottom_ratio=1.0):
    h, w = image.shape[:2]
    top = int(h * top_ratio)
    bottom = int(h * bottom_ratio)
    return image[top:bottom, :]


def is_dialog_open_by_template(image, threshold=0.5, debug=False):
    dialog_region = extract_dialog_region(image)
    dialog_gray = cv2.cvtColor(dialog_region, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(
        dialog_gray, dialog_template_gray, cv2.TM_CCOEFF_NORMED)
    max_val = np.max(res)

    if debug:
        print(f"[DEBUG] Coincidencia con cuadro de texto: {max_val:.2f}")

    return max_val >= threshold
