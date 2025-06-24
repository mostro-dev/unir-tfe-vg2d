from game_agent.dqn.environment import GameEnvironment
import time
from dotenv import load_dotenv
from game_agent.controller.keyboard_controller import move
from openai import OpenAI
import json
import os
import re

# Dotenv Config
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
Eres un asistente para controlar un personaje en Pokémon Rojo.
Tu tarea es interpretar instrucciones en lenguaje natural y convertirlas en una lista de movimientos simples.

Solo puedes usar las siguientes direcciones:
- 'up'
- 'down'
- 'left'
- 'right'

Tu respuesta debe ser una lista JSON como esta:
[
  {"direction": "right", "steps": 1},
  {"direction": "up", "steps": 5}
]
"""

GPT_MODEL = "gpt-3.5-turbo"  # Cambia a "gpt-4" si tienes acceso


def extract_json_array(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No se encontró un array JSON en la respuesta.")


def ask_llm(user_command: str) -> list[dict]:
    """Pide al LLM que transforme NL en JSON de movimientos."""
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_command}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content
    return extract_json_array(content)


def process_command_with_llm(user_command):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_command}
        ],
        temperature=0
    )

    content = response.choices[0].message.content
    try:
        instructions = extract_json_array(content)
        for action in instructions:
            move(action["direction"], action["steps"])
            time.sleep(0.3)  # Espera un poco entre movimientos
    except Exception as e:
        print("[ERROR] No se pudo interpretar la respuesta del modelo:", e)
        print("Respuesta del modelo:", content)


# ─── Control Directo: JSON → move() ───────────────────────────────────────────
def direct_control(user_command: str, delay: float = 0.3):
    """
    Interpreta el comando con el LLM y ejecuta move(direction, steps).
    No usa world_map ni step().
    """
    try:
        instructions = ask_llm(user_command)
        for instr in instructions:
            direction = instr["direction"]
            steps = instr["steps"]
            for _ in range(steps):
                move(direction)
                time.sleep(delay)
    except Exception as e:
        print("[ERROR][direct_control] ", e)


# ─── Control Basado en Mapa: JSON → env.step() ───────────────────────────────
_env: GameEnvironment | None = None


def map_control(user_command: str, delay: float = 0.3):
    """
    Interpreta el comando con el LLM y ejecuta env.step(direction) por paso,
    aprovechando world_map, recompensas y colisiones.
    """
    global _env
    if _env is None:
        _env = GameEnvironment(save_mode=False, punish_revisit=True)

    try:
        instructions = ask_llm(user_command)
        for instr in instructions:
            dir_ = instr["direction"]
            steps = instr["steps"]
            for _ in range(steps):
                state, reward, done = _env.step(dir_, debug=True)
                print(f"[map_control] → {dir_}, reward={reward:.2f}")
                time.sleep(delay)
    except Exception as e:
        print("[ERROR][map_control] ", e)


# ─── Ejemplo de uso ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chat Controller Ready. Escribe tu comando en NL (o 'exit').")
    while True:
        cmd = input("> ")
        if cmd.lower() in ("exit", "quit"):
            break

        print("\n== Control Directo ==")
        direct_control(cmd)

        print("\n== Control Basado en Mapa ==")
        map_control(cmd)
        print("\n" + "="*50 + "\n")
