
from enum import Enum

# VisualBoyAdvance window region config (macOS)
y_offset = 55
x_offset = 64
window_width = 480
window_height = 370

calc_window_width = window_width - (x_offset * 2)
calc_window_height = window_height - y_offset

GAME_REGION = (x_offset, y_offset, calc_window_width, calc_window_height)

TILE_HEIGHT = 35
TILE_WIDTH = 35

WHITE_THRESHOLD = 215
OBSTACLE_THRESHOLD = 200

PLAYER_POSITION = (4, 4)

DIRECTIONS = ['up', 'right', 'down', 'left']
DIRECTIONS_TUPLE = ('up', 'right', 'down', 'left')


class TileType(str, Enum):
    FLOOR = "FLOOR"
    WALL = "WALL"
    INFO = "INFO"
    DOOR = "DOOR"
    UNKNOWN = "UNKNOWN"
