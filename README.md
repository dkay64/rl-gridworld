# RL GridWorld with Tabular Q-learning

This project trains a simple tabular Q-learning agent on a 4x4 GridWorld. The agent learns to navigate from the top-left start state to the bottom-right goal by balancing exploration and exploitation. The repository includes training logs, learned value visualizations, and sample outputs for inspection.

## Setup

```bash
conda create -n cuda_env python=3.10 -y
conda activate cuda_env
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Explanation

- **Environment:** `envs/gridworld.py` defines a deterministic 4x4 grid with four actions (up, right, down, left), a start state at cell 0, and a goal at cell 15 that yields a reward of `+1` and terminates the episode. All other transitions incur a small penalty of `-0.01` to encourage short paths.
- **Algorithm:** `agents/q_learning_agent.py` implements an epsilon-greedy tabular Q-learning agent that updates its Q-table after every transition using learning rate `alpha`, discount factor `gamma`, and exploration rate `epsilon`. `main.py` orchestrates training, prints rolling-average returns, and evaluates the greedy policy after training.
- **Visualization:** Running `main.py` produces Matplotlib plots. `QLearningOnGridWorldFigure*.png` capture the learning curve (episode return vs. episode index) and the learned state-value heatmap; add updated screenshots here when you regenerate results.
- **Outputs:** `QLearningOnGridWorldOutput*.txt` show sample logs, including training progress and a greedy trajectory that reaches the goal with high reward.
