import heapq
import random
from collections import Counter

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, walls, width, height):
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < height and 0 <= ny < width and (nx, ny) not in walls:
            yield (nx, ny)

def astar(start, goal, walls, width, height):
    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start, goal), 0, start, [start]))
    closed = set()
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in closed:
            continue
        closed.add(current)
        for neighbor in get_neighbors(current, walls, width, height):
            if neighbor in closed:
                continue
            new_g = g + 1
            new_f = new_g + manhattan_distance(neighbor, goal)
            heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

    return []

def compute_best_trap_tile(env):
    score = Counter()
    thief_pos = env.thief_pos
    exit_pos  = env.exit
    walls, width, height = env.walls, env.width, env.height
    for g in env.gems:
        path = astar(thief_pos, g, walls, width, height)
        for tile in path[1:]:
            score[tile] += 1
    for g in list(env.gems) + env.collected:
        path = astar(g, exit_pos, walls, width, height)
        for tile in path[1:]:
            score[tile] += 2
    candidates = [
        (tile, sc) for tile, sc in score.items()
        if env._is_valid(tile)
           and tile not in env.alarms
           and tile not in env.traps
    ]
    if not candidates:
        return None
    best_tile, _ = max(candidates, key=lambda x: x[1])
    return best_tile

def state_to_key(state):
    thief_pos, guard_pos, gems, traps, alarm, exit_pos = state
    parts = [
        f"T{thief_pos[0]},{thief_pos[1]}",
        f"G{guard_pos[0]},{guard_pos[1]}"
    ]
    for g in gems:
        parts.append(f"D{g[0]},{g[1]}")
    for t in traps:
        parts.append(f"X{t[0]},{t[1]}")
    parts.append(f"A{int(alarm)}")
    parts.append(f"E{exit_pos[0]},{exit_pos[1]}")
    return '|'.join(parts)
