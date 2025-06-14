import cv2
import numpy as np
import os

# Ruta base donde están los templates
TEMPLATE_DIR = "./game_agent/vision/templates"

# Cargar manualmente las 4 plantillas
template_paths = [
    os.path.join(TEMPLATE_DIR, "oak_template_1.png"),
    os.path.join(TEMPLATE_DIR, "oak_template_2.png"),
    os.path.join(TEMPLATE_DIR, "oak_template_3.png"),
    os.path.join(TEMPLATE_DIR, "oak_template_4.png"),
]

oak_templates = []
for path in template_paths:
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"No se pudo cargar la plantilla: {path}")
    template = template.astype(np.uint8)  # Aseguramos tipo uint8
    oak_templates.append(template)


def is_oak_zone_triggered(image, threshold=0.85, debug=False):
    """
    Evalúa si el personaje se encuentra en la zona que activa el evento del Profesor Oak.

    Parámetros:
        image: imagen en escala de grises del juego procesada (post-pipeline)
        threshold: valor mínimo de coincidencia con la plantilla
        debug: si es True, imprime valores de coincidencia

    Retorna:
        True si se detecta alguna coincidencia, False en caso contrario.
    """
    # Asegurar que la imagen esté en escala de grises
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Asegurar que la imagen sea tipo uint8
    image = image.astype(np.uint8)

    for idx, template in enumerate(oak_templates):
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(res)

        if debug:
            print(
                f"[DEBUG] Coincidencia con oak_template_{idx+1}: {max_val:.2f}")

        if max_val >= threshold:
            return True

    return False
