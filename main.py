# main.py
from config import GAME_REGION
from game_agent.llm_interface.chat_controller import process_command_with_llm
from game_agent.vision.screen_reader import capture_region, save_image
import time
from game_agent.controller.keyboard_controller import press, move

# Move test 1
print("Starting in 3 seconds...")
time.sleep(3)

# LLM Command Execution
# while True:
#     cmd = input("¿Qué debe hacer el personaje? ('salir' para terminar): ")
#     if cmd.lower() == "salir":
#         break
#     try:
#         print("Waiting 3 seconds...")
#         time.sleep(3)
#         process_command_with_llm(cmd)
#     except Exception as e:
#         print(f"Error: {e}")

# Image Capture
# frame = capture_region(GAME_REGION)
# save_image(frame)
