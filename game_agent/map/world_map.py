
import json
from collections import defaultdict
import os


class WorldMap:
    def __init__(self, save_path="game_agent/map/world_map.json"):
        self.position = (0, 0)  # posición actual (x, y)
        self.map_data = defaultdict(
            lambda: defaultdict(float))  # (x, y) -> {type: prob}
        self.visited = set()
        self.save_path = save_path

        if os.path.exists(save_path):
            self.load()

    def move(self, direction):
        x, y = self.position
        if direction == "up":
            self.position = (x, y - 1)
        elif direction == "down":
            self.position = (x, y + 1)
        elif direction == "left":
            self.position = (x - 1, y)
        elif direction == "right":
            self.position = (x + 1, y)
        # Z no cambia posición

    def update_tile(self, label, prob=1.0):
        """
        label: 'FLOOR', 'WALL', 'INFO', 'DOOR', etc.
        Actualiza la probabilidad de que el tile actual tenga cierta etiqueta.
        """
        tile = self.position
        self.map_data[tile][label] += prob
        self.normalize_tile(tile)
        self.visited.add(tile)

    def normalize_tile(self, tile):
        total = sum(self.map_data[tile].values())
        if total > 0:
            for key in self.map_data[tile]:
                self.map_data[tile][key] /= total

    def get_tile_info(self, x, y):
        return dict(self.map_data.get((x, y), {}))

    def is_visited(self, x, y):
        return (x, y) in self.visited

    def save(self):
        serializable_map = {
            f"{x},{y}": data for (x, y), data in self.map_data.items()
        }
        with open(self.save_path, "w") as f:
            json.dump({
                "position": self.position,
                "map": serializable_map,
                "visited": list(map(list, self.visited))
            }, f, indent=2)

    def load(self):
        with open(self.save_path, "r") as f:
            data = json.load(f)
            self.position = tuple(data["position"])
            self.map_data = defaultdict(lambda: defaultdict(float), {
                tuple(map(int, k.split(","))): defaultdict(float, v)
                for k, v in data["map"].items()
            })
            self.visited = set(tuple(v) for v in data["visited"])
