import random

class HeistEnv:
    ACTIONS = list(range(6))

    def __init__(self):
        self.height, self.width = 6, 6
        self._corners = [(0, 0), (0, 5), (5, 0), (5, 5)]
        self.walls = {(1, 2), (2, 3), (3, 1), (4, 4)}
        self.alarms = {(2, 2), (3, 3)}
        self.TRAP_TTL = 10
        self.EXIT_CHANGE_INTERVAL = 20
        self.reset()

    def reset(self):
        self.global_step_count = 0
        # Agent start positions
        self.thief_pos = (0, 0)
        self.guard_pos = (5, 5)
        possible_exits = [c for c in self._corners if c not in [(0, 0), (5, 5)]]
        self.exit = random.choice(possible_exits)
        forbidden = set(self.walls) | set(self.alarms) | {(0, 0), (5, 5), self.exit}
        empties = [(x, y) for x in range(self.height) for y in range(self.width)
                   if (x, y) not in forbidden]
        self.gems = set(random.sample(empties, 2))
        self.traps = set()
        self.trap_timers = {}
        self.collected = []
        self.alarm_triggered = False
        self.alarm_timer = 0
        self.done = False
        self.guard_visited = {self.guard_pos}
        self.last_guard_pos = None
        self.guard_idle_steps = 0
        return self._get_state()

    def step(self, thief_action, guard_action):
        if self.done:
            raise RuntimeError("Episode has terminated; call reset().")
        self.global_step_count += 1
        if self.global_step_count % self.EXIT_CHANGE_INTERVAL == 0:
            candidates = [c for c in self._corners if c != self.exit]
            self.exit = random.choice(candidates)
        r_thief, r_guard = 0.0, 0.0
        old_thief = self.thief_pos
        old_guard = self.guard_pos
        for pos in list(self.trap_timers):
            self.trap_timers[pos] -= 1
            if self.trap_timers[pos] <= 0:
                del self.trap_timers[pos]
                if pos in self.traps:
                    self.traps.remove(pos)
                r_guard -= 0.5
        self._apply_action('thief', thief_action)
        if self.thief_pos == old_thief:
            r_thief -= 0.1
        from utils import manhattan_distance
        if self.gems:
            goal = min(self.gems, key=lambda g: manhattan_distance(old_thief, g))
        else:
            goal = self.exit
        d_old = manhattan_distance(old_thief, goal)
        d_new = manhattan_distance(self.thief_pos, goal)
        beta_t = 0.05
        r_thief += beta_t * (d_old - d_new)
        self._apply_action('guard', guard_action)
        if self.guard_pos == old_guard and self.guard_pos in self.gems:
            r_guard -= 0.2
        d_old_g = manhattan_distance(old_guard, self.thief_pos)
        d_new_g = manhattan_distance(self.guard_pos, self.thief_pos)
        beta_g = 0.15
        r_guard += beta_g * (d_old_g - d_new_g)
        if self.guard_pos not in self.guard_visited:
            r_guard += 0.1
            self.guard_visited.add(self.guard_pos)
        if self.last_guard_pos == self.guard_pos:
            self.guard_idle_steps += 1
        else:
            self.guard_idle_steps = 0
        self.last_guard_pos = self.guard_pos
        if self.guard_idle_steps > 3:
            r_guard -= 0.1
        if self.thief_pos in self.alarms:
            self.alarm_triggered = True
            self.alarm_timer = 3
            r_thief -= 1.0
            r_guard += 1.0
        if self.thief_pos in self.gems:
            self.gems.remove(self.thief_pos)
            self.collected.append(self.thief_pos)
            r_thief += 1.0
        if self.thief_pos in self.traps:
            self.traps.remove(self.thief_pos)
            if self.thief_pos in self.trap_timers:
                del self.trap_timers[self.thief_pos]
            r_thief -= 2.0
            r_guard += 2.0
        result = None
        if self.thief_pos == self.guard_pos:
            r_thief -= 5.0
            r_guard += 5.0
            self.done = True
            result = 'guard'
        elif len(self.collected) == 2 and self.thief_pos == self.exit:
            r_thief += 5.0
            r_guard -= 5.0
            self.done = True
            result = 'thief'
        if self.alarm_timer > 0:
            self.alarm_timer -= 1
            if self.alarm_timer == 0:
                self.alarm_triggered = False
        state = self._get_state()
        info = {'result': result}
        return state, (r_thief, r_guard), self.done, info

    def _apply_action(self, agent, action):
        deltas = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        if agent == 'thief':
            if action == 5:
                return
            dx, dy = deltas.get(action, (0, 0))
            new_pos = (self.thief_pos[0] + dx, self.thief_pos[1] + dy)
            if self._is_valid(new_pos):
                self.thief_pos = new_pos
        else:
            if action == 5:
                if len(self.traps) < 2:
                    from utils import compute_best_trap_tile
                    target = compute_best_trap_tile(self) or self.guard_pos
                    self.traps.add(target)
                    self.trap_timers[target] = self.TRAP_TTL
                return
            dx, dy = deltas.get(action, (0, 0))
            new_pos = (self.guard_pos[0] + dx, self.guard_pos[1] + dy)
            if self._is_valid(new_pos):
                self.guard_pos = new_pos

    def _is_valid(self, pos):
        x, y = pos
        if not (0 <= x < self.height and 0 <= y < self.width):
            return False
        if pos in self.walls:
            return False
        return True

    def _get_state(self):
        return (
            self.thief_pos,
            self.guard_pos,
            tuple(sorted(self.gems)),
            tuple(sorted(self.traps)),
            self.alarm_triggered,
            self.exit,
        )

    def render_ascii(self):
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for (x, y) in self.walls:
            grid[x][y] = '#'
        for (x, y) in self.alarms:
            grid[x][y] = 'A'
        for (x, y) in self.gems:
            grid[x][y] = 'D'
        for (x, y) in self.traps:
            grid[x][y] = 'X'
        ex, ey = self.exit
        grid[ex][ey] = 'E'
        tx, ty = self.thief_pos
        grid[tx][ty] = 'T'
        gx, gy = self.guard_pos
        grid[gx][gy] = 'G'
        print("\n".join(" ".join(row) for row in grid))
