import pyautogui
import time

def press(key, duration=0.2):
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)
    time.sleep(0.1)

def move(direction, steps=1):
    for _ in range(steps):
        press(direction)