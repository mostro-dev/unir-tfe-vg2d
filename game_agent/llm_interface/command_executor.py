# [DEPRECATED]
import re
from game_agent.controller.keyboard_controller import move

DIRECTIONS = {
    "arriba": "up",
    "abajo": "down",
    "izquierda": "left",
    "derecha": "right"
}

def parse_command(text):
    """
    Convierte un comando en lenguaje natural en una dirección y pasos.
    Ej: "ve abajo", "camina a la izquierda 3 veces"
    """
    text = text.lower()

    # Buscar dirección
    direction = None
    for word in DIRECTIONS:
        if word in text:
            direction = DIRECTIONS[word]
            break
    if not direction:
        raise ValueError("No se pudo entender la dirección en el comando.")

    # Buscar cantidad de pasos (opcional)
    match = re.search(r"(\d+)", text)
    steps = int(match.group(1)) if match else 1

    return direction, steps


def execute_command(text):
    """
    Ejecuta el movimiento basado en un comando de texto.
    """
    direction, steps = parse_command(text)
    move(direction, steps)
