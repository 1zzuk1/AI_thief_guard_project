import os
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'guard_agent.pkl')

with open(MODEL_PATH, 'rb') as f:
    agent = pickle.load(f)

q_table = agent.q_table
rows = []
for state, q_vals in q_table.items():
    thief_view, guard_pos, gems, traps, alarm, exit_pos = state
    if thief_view is None:
        tvx, tvy = -1, -1
    else:
        tvx, tvy = thief_view
    gx, gy = guard_pos
    gem_list = list(gems)
    while len(gem_list) < 2:
        gem_list.append((-1, -1))
    (g0x, g0y), (g1x, g1y) = gem_list
    trap_list = list(traps)
    while len(trap_list) < 2:
        trap_list.append((-1, -1))
    (t0x, t0y), (t1x, t1y) = trap_list
    ex, ey = exit_pos
    best_action = int(q_vals.index(max(q_vals)))

    rows.append({
        'thief_view_x': tvx, 'thief_view_y': tvy,
        'guard_x': gx,       'guard_y': gy,
        'alarm': int(alarm),
        'exit_x': ex,        'exit_y': ey,
        'gem0_x': g0x,       'gem0_y': g0y,
        'gem1_x': g1x,       'gem1_y': g1y,
        'trap0_x': t0x,      'trap0_y': t0y,
        'trap1_x': t1x,      'trap1_y': t1y,
        'action': best_action
    })

df = pd.DataFrame(rows)
feature_cols = [c for c in df.columns if c != 'action']
X = df[feature_cols]
y = df['action']
clf = DecisionTreeClassifier(max_depth=7)
clf.fit(X, y)
print(export_text(clf, feature_names=feature_cols))
