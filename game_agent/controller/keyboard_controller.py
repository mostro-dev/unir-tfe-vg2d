import pyautogui
import time


def press(key, duration=0.1):
    print(f"\nPressed {key} Button")
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)
    time.sleep(0.1)


def move(direction, steps=1):
    press(direction)
    # for _ in range(steps):
    #     press(direction)
