# file: evaluate.py
import os
import argparse
import random

from env.heist_env import HeistEnv
from agents.thief_agent import ThiefAgent
from agents.guard_agent import GuardAgent
from utils import manhattan_distance

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate agents in the Heist environment with partial observability for the guard."
    )
    parser.add_argument(
        '--role', choices=['thief', 'guard', 'both'], default='both',
        help="Which agent(s) to evaluate: 'thief', 'guard', or 'both'."
    )
    parser.add_argument(
        '--episodes', type=int, default=1000,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--max_steps', type=int, default=50,
        help='Max steps per episode'
    )
    parser.add_argument(
        '--model_dir', type=str, default='models',
        help='Directory where trained agents are saved'
    )
    parser.add_argument(
        '--render', action='store_true',
        help='Render each episode in ASCII'
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
    return RandomAgent(action_space)

def load_agent(role, action_space, model_dir):
    if role == 'thief':
        path = os.path.join(model_dir, 'thief_agent.pkl')
        if os.path.exists(path):
            return ThiefAgent.load(path)
    elif role == 'guard':
        path = os.path.join(model_dir, 'guard_agent.pkl')
        if os.path.exists(path):
            return GuardAgent.load(path)
    return make_random_agent(action_space)

def evaluate():
    args = parse_args()
    env = HeistEnv()
    action_space = env.ACTIONS

    thief_agent = load_agent('thief', action_space, args.model_dir)
    guard_agent = load_agent('guard', action_space, args.model_dir)

    stats = {'thief_wins': 0, 'guard_wins': 0, 'draws': 0, 'steps': []}

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state_thief, state_guard = split_state(obs)
        done = False
        step = 0

        if args.render:
            print(f"\nEpisode {ep}")
            env.render_ascii()
            print()

        while not done and step < args.max_steps:
            # Select actions
            a_thief = thief_agent.select_action(state_thief) if args.role in ('thief','both') \
                      else random.choice(action_space)
            a_guard = guard_agent.select_action(state_guard) if args.role in ('guard','both') \
                      else random.choice(action_space)

            obs, (r_thief, r_guard), done, info = env.step(a_thief, a_guard)
            next_thief, next_guard = split_state(obs)
            step += 1

            if args.render:
                env.render_ascii()
                print()
            state_thief, state_guard = next_thief, next_guard

        result = info.get('result')
        if result == 'thief':
            stats['thief_wins'] += 1
        elif result == 'guard':
            stats['guard_wins'] += 1
        else:
            stats['draws'] += 1
        stats['steps'].append(step)

    total = args.episodes
    t = stats['thief_wins']
    g = stats['guard_wins']
    d = stats['draws']
    avg_steps = sum(stats['steps']) / len(stats['steps']) if stats['steps'] else 0

    print("\n=== Evaluation Results ===")
    print(f"Thief wins: {t}/{total} ({t/total*100:.1f}%)")
    print(f"Guard wins: {g}/{total} ({g/total*100:.1f}%)")
    print(f"Draws     : {d}/{total} ({d/total*100:.1f}%)")
    print(f"Avg steps : {avg_steps:.1f}")

if __name__ == '__main__':
    evaluate()
