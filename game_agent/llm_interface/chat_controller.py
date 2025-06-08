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

GPT_MODEL = "gpt-3.5-turbo" # Cambia a "gpt-4" si tienes acceso

def extract_json_array(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No se encontró un array JSON en la respuesta.")

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
