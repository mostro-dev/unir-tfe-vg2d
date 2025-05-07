# main.py
import time
from game_agent.controller.keyboard_controller import press, move

# Move test 1
print("Starting in 3 seconds...")
time.sleep(3)

# Move down 5 times
move('right', steps=1)
move('up', steps=5)
move('right', steps=2)

press('z')
