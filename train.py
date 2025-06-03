import os
import argparse
import random

from env.heist_env import HeistEnv
from agents.thief_agent import ThiefAgent
from agents.guard_agent import GuardAgent
from utils import manhattan_distance

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train agents in the Heist environment with partial observability for the guard."
    )
    parser.add_argument(
        '--role', choices=['thief', 'guard', 'both'], default='both',
        help="Which agent(s) to train: 'thief', 'guard', or 'both'."
    )
    parser.add_argument(
        '--episodes', type=int, default=50000,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--max_steps', type=int, default=50,
        help='Max steps per episode'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--epsilon', type=float, default=0.1,
        help='Exploration rate'
    )
    parser.add_argument(
        '--save_dir', type=str, default='models',
        help='Directory to save trained agents'
    )
    return parser.parse_args()

def mask_guard_state(tuple_state):
    thief_pos, guard_pos, gems, traps, alarm, exit_pos = tuple_state
    if not alarm and manhattan_distance(thief_pos, guard_pos) > 2:
        thief_view = None
    else:
        thief_view = thief_pos
    return (thief_view, guard_pos, gems, traps, alarm, exit_pos)

def split_state(global_state):
    if isinstance(global_state, dict):
        return global_state['thief'], global_state['guard']
    else:
        return global_state, mask_guard_state(global_state)

def make_random_agent(action_space):
    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space
        def select_action(self, state):
            return random.choice(self.action_space)
        def update(self, *args, **kwargs):
            pass
        def save(self, filepath):
            pass
    return RandomAgent(action_space)

def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = HeistEnv()
    action_space = env.ACTIONS

    thief_agent = ThiefAgent(
        action_space,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon
    )
    guard_agent = GuardAgent(
        action_space,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon
    )
    random_agent = make_random_agent(action_space)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state_thief, state_guard = split_state(obs)
        done = False
        step = 0

        while not done and step < args.max_steps:
            if args.role in ('thief', 'both'):
                a_thief = thief_agent.select_action(state_thief)
            else:
                a_thief = random_agent.select_action(state_thief)
            if args.role in ('guard', 'both'):
                a_guard = guard_agent.select_action(state_guard)
            else:
                a_guard = random_agent.select_action(state_guard)
            next_obs, (r_thief, r_guard), done, info = env.step(a_thief, a_guard)
            next_thief, next_guard = split_state(next_obs)
            if args.role in ('thief', 'both'):
                thief_agent.update(state_thief, a_thief, r_thief, next_thief, done)
            if args.role in ('guard', 'both'):
                guard_agent.update(state_guard, a_guard, r_guard, next_guard, done)
            state_thief, state_guard = next_thief, next_guard
            step += 1
        if ep % 1000 == 0:
            print(f"Episode {ep}/{args.episodes} ended: result={info.get('result')}")
    if args.role in ('thief', 'both'):
        thief_agent.save(os.path.join(args.save_dir, 'thief_agent.pkl'))
    if args.role in ('guard', 'both'):
        guard_agent.save(os.path.join(args.save_dir, 'guard_agent.pkl'))

    print(f"Training complete. Models saved to '{args.save_dir}'.")

if __name__ == '__main__':
    train()
