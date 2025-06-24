# game_agent/map/world_map.py

import json
import os
from config import TileType


class WorldMap:
    def __init__(self, path="game_agent/map/world_map.json", debug: bool = False):
        self.path = path
        self.debug = debug
        # { (x,y): { TileType.VALUE: prob, …, "_counts": {TileType.VALUE: n, …}, "_visits": n } }
        self.map = {
            (0, 0): {
                TileType.FLOOR.value: 1.0,
                "_counts": {TileType.FLOOR.value: 1},
                "_visits": 1
            }
        }

    def update_tile(self, coord, tile_type: TileType):
        entry = self.map.setdefault(coord, {})
        # 1) Asegurar diccionario de conteos
        counts = entry.setdefault("_counts", {})
        key = tile_type.value
        counts[key] = counts.get(key, 0) + 1

        # 2) Recalcular probabilidades
        total = sum(counts.values())
        for t, c in counts.items():
            entry[t] = c / total

        if self.debug:
            probs = ", ".join(f"{t}={entry[t]:.2f}" for t in counts)
            print(f"[DEBUG][WorldMap] update_tile {coord} → {probs}")

    def mark_visited(self, coord):
        entry = self.map.setdefault(coord, {})
        entry["_visits"] = entry.get("_visits", 0) + 1
        if self.debug:
            print(
                f"[DEBUG][WorldMap] mark_visited {coord} → visitas={entry['_visits']}")

    def save(self):
        serializable = {
            f"{x},{y}": {
                **{k: v for k, v in entry.items()},
                # opcionalmente no vuelcas _counts al JSON
                # entry
            }
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
