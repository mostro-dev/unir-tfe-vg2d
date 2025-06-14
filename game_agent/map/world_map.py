# game_agent/map/world_map.py

import json
import os
from config import TileType


class WorldMap:
    def __init__(self, path="game_agent/map/world_map.json", debug=False):
        self.path = path
        self.debug = debug
        # { (x,y): { "FLOOR": prob, … , "_visits": n } }
        self.map = {(0, 0): {TileType.FLOOR.value: 1.0, "_visits": 0}}

    def update_tile(self, coord, tile_type: TileType, prob: float):
        entry = self.map.setdefault(coord, {})
        entry[tile_type.value] = prob
        if self.debug:
            print(
                f"[DEBUG][WorldMap] update_tile {coord} → {tile_type.value} (p={prob:.2f})")

    def mark_visited(self, coord):
        entry = self.map.setdefault(coord, {})
        entry["_visits"] = entry.get("_visits", 0) + 1
        if self.debug:
            print(
                f"[DEBUG][WorldMap] mark_visited {coord} → visitas={entry['_visits']}")

    def save(self):
        # Convertir claves tuple → "x,y" para JSON
        serializable = {
            f"{x},{y}": entry
            for (x, y), entry in self.map.items()
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(serializable, f, indent=2)
        if self.debug:
            print(f"[DEBUG][WorldMap] guardando mapa en {self.path}")

    def load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            data = json.load(f)
        self.map = {}
        for key, entry in data.items():
            x_str, y_str = key.split(",")
            coord = (int(x_str), int(y_str))
            self.map[coord] = entry
        if self.debug:
            print(f"[DEBUG][WorldMap] cargado desde {self.path}")
