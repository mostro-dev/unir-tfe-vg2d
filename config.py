# VisualBoyAdvance window region config (macOS)
y_offset = 55
x_offset = 64
window_width = 480
window_height = 370

calc_window_width = window_width - (x_offset * 2)
calc_window_height = window_height - y_offset

GAME_REGION = (x_offset, y_offset, calc_window_width, calc_window_height)

TILE_HEIGHT = 35
TILE_WIDTH = 32
