import os
import sys
import random
import pygame

from env.heist_env import HeistEnv
from agents.thief_agent import ThiefAgent
from agents.guard_agent import GuardAgent
from utils import manhattan_distance

# ---- Configuration ----
CELL_SIZE = 80
GRID_SIZE = 6
WIDTH = CELL_SIZE * GRID_SIZE
HEIGHT = CELL_SIZE * GRID_SIZE
FPS = 3   # frames per second => ~0.33s per step
MODEL_DIR = 'models'

# Colors
WHITE  = (255,255,255)
GRAY   = (200,200,200)
BLACK  = (0,0,0)
RED    = (255,0,0)
YELLOW = (255,255,0)
ORANGE = (255,165,0)
GREEN  = (0,255,0)
BLUE   = (0,0,255)
PURPLE = (128,0,128)


def mask_guard_state(tuple_state):
    thief_pos, guard_pos, gems, traps, alarm, exit_pos = tuple_state
    if not alarm and manhattan_distance(thief_pos, guard_pos) > 2:
        thief_view = None
    else:
        thief_view = thief_pos
    return (thief_view, guard_pos, gems, traps, alarm, exit_pos)


def split_state(obs):
    if isinstance(obs, dict):
        return obs['thief'], obs['guard']
    else:
        return obs, mask_guard_state(obs)


def load_agent(agent_cls, filename, action_space):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        try:
            return agent_cls.load(path)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    print(f"Using random agent for {filename}")
    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space
        def select_action(self, state):
            return random.choice(self.action_space)
    return RandomAgent(action_space)


def draw_grid(screen, env):
    """Draw walls, alarms, gems, traps, exit, and agents."""
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)

            pos = (x,y)
            if pos in env.walls:
                pygame.draw.rect(screen, BLACK, rect)
            elif pos in env.alarms:
                pygame.draw.rect(screen, RED, rect)
            if pos in env.gems:
                pygame.draw.circle(screen, YELLOW, rect.center, CELL_SIZE//4)
            if pos in env.traps:
                pygame.draw.rect(screen, ORANGE, rect)
            if pos == env.exit:
                pygame.draw.rect(screen, GREEN, rect)

    # Draw thief
    tx, ty = env.thief_pos
    trect = pygame.Rect(ty*CELL_SIZE, tx*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.circle(screen, BLUE, trect.center, CELL_SIZE//3)
    # Draw guard
    gx, gy = env.guard_pos
    grect = pygame.Rect(gy*CELL_SIZE, gx*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.circle(screen, PURPLE, grect.center, CELL_SIZE//3)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("HeistEnv Visualization")
    clock = pygame.time.Clock()

    env = HeistEnv()
    action_space = env.ACTIONS

    thief_agent = load_agent(ThiefAgent, 'thief_agent.pkl', action_space)
    guard_agent = load_agent(GuardAgent, 'guard_agent.pkl', action_space)

    running = True
    while running:
        obs = env.reset()
        state_thief, state_guard = split_state(obs)
        done = False

        while not done and running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

            a_thief = thief_agent.select_action(state_thief)
            a_guard = guard_agent.select_action(state_guard)

            new_obs, (r_t, r_g), done, info = env.step(a_thief, a_guard)
            next_thief, next_guard = split_state(new_obs)

            draw_grid(screen, env)
            pygame.display.flip()
            clock.tick(FPS)

            # Advance states
            state_thief, state_guard = next_thief, next_guard

        if not running:
            break

        print(f"Episode ended: {info.get('result')}")
        pygame.time.delay(1000)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
