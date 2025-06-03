# README

This repository contains several trained Q-learning agents for the Heist environment. Only the top six agents listed below are recommended for evaluation and further testing. The remaining files illustrate earlier or alternative training approaches and can serve as examples of poorly trained agents or experiments in modified environments.

---

## Recommended Agents (Worth Testing)

1. **guard\_agent.pkl**

   * Guard agent trained with the standard setup (including anti-camping penalties and changing exit).
   * Best-performing guard policy under our primary training regime.

2. **thief\_agent.pkl**

   * Thief agent trained with the standard setup (including potential-based shaping toward gems/exit).
   * Best-performing thief policy under our primary training regime.

3. **guard\_agent\_no\_camping.pkl**

   * Guard agent trained **without** the idle‐camping penalty.
   * Illustrates the effect of removing the anti‐camping reward structure.

4. **thief\_agent\_no\_camping.pkl**

   * Thief agent trained **without** the idle‐camping penalty on the guard.
   * Illustrates how the thief’s policy changes when the guard is not penalized for camping.

5. **best\_guard\_so\_far.pkl**

   * Snapshot of the best guard policy obtained during training (before final tuning).
   * Useful for comparing intermediate performance to `guard_agent.pkl`.

6. **best\_thief\_so\_far.pkl**

   * Snapshot of the best thief policy obtained during training (before final tuning).
   * Useful for comparing intermediate performance to `thief_agent.pkl`.

---

## Other Agent Files (Illustrative / Poorly Trained Examples)

The following files represent agents trained under different conditions or with suboptimal hyperparameters. They are **not** recommended for primary evaluation but can be used to study:

* The impact of alternative training schedules (e.g., “solo” vs. “versus” training)

* Effects of removing randomness, camping penalties, or changing trap strategies

* Artifacts of incorrect reward‐shaping or environment modifications

* **guard\_weirdly\_trained\_agent.pkl**

* **thief\_weirdly\_trained\_on\_tra.pkl**

* **guard\_agentVS\_with\_no\_randomness.pkl**

* **thief\_agentVS\_with\_no\_randomness.pkl**

* **guard\_agent\_solo\_trained.pkl**

* **thief\_agent\_solo\_trained.pkl**

---

## Usage

1. **Loading a Bundle**
   Each `.pkl` file is an `AgentBundle` containing both the agent instance and metadata (hyperparameters, training details). To load:

   ```python
   from agents.bundle import AgentBundle

   bundle = AgentBundle.load("models/guard_agent.pkl")
   guard_agent = bundle.get_agent()
   metadata    = bundle.get_metadata()
   ```

2. **Evaluating an Agent**
   Use the `evaluate.py` script to measure win rates and average episode lengths. For example:

   ```bash
   python evaluate.py --role both --episodes 1000 --max_steps 200
   ```

   * `--role` can be `thief`, `guard`, or `both`.
   * `--episodes` specifies how many evaluation episodes to run.
   * `--max_steps` caps the number of steps per episode.

3. **Visualizing a Run**
   Launch `visualize.py` to see a Pygame display of agent behavior:

   ```bash
   python visualize.py
   ```

---

## Directory Structure

```
/models
├── guard_agent.pkl
├── thief_agent.pkl
├── guard_agent_no_camping.pkl
├── thief_agent_no_camping.pkl
├── best_guard_so_far.pkl
├── best_thief_so_far.pkl
├── guard_weirdly_trained_agent.pkl
├── thief_weirdly_trained_on_tra.pkl
├── guard_agentVS_with_no_randomness.pkl
├── thief_agentVS_with_no_randomness.pkl
├── guard_agent_solo_trained.pkl
└── thief_agent_solo_trained.pkl
```

* Top six files (first two rows) are recommended for benchmarking.
* Remaining files illustrate edge cases or experimental setups.

---

## Notes

* **Anti-Camping Penalty**
  During training, we added a small negative reward when the guard stayed in the same position for more than three consecutive steps. Removing this penalty (see `*_no_camping.pkl` files) results in guard policies that tend to “camp” on high-value tiles (e.g., on gems), making them easier to predict.

* **Dynamic Exit**
  The exit relocates every 20 steps to prevent overfitting to static corner locations. This encourages more robust thief strategies.

* **Potential-Based Shaping**
  The thief receives a small positive shaping reward proportional to the reduction in Manhattan distance to the nearest gem (or, once gems are collected, to the exit). This speeds up learning by biasing exploration toward valuable goals.

Feel free to explore, compare, and build upon these models!
