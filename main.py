# main.py
from game_agent.vision.screen_reader import capture_region, save_image
import time
from game_agent.controller.keyboard_controller import press, move

# Move test 1
print("Starting in 3 seconds...")
time.sleep(3)

# Move down 5 times
# move('right', steps=1)
# move('up', steps=5)
# move('right', steps=2)

# press('z')

# Image capture test

# MacOS Screen Config with VisualBoy
# Comment this if needed or using another config
y_offset = 55
x_offset = 64
window_width = 480
window_height = 370
calc_window_width = window_width - (x_offset * 2)
calc_window_height = window_height - y_offset

print(
    f"Window size: [Width]: {calc_window_width} [height]: {calc_window_height}")
region = (x_offset, y_offset, calc_window_width, calc_window_height)

frame = capture_region(region)
save_image(frame)
