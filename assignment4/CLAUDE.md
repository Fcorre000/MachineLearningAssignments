# Assignment 4: Reinforcement Learning

## Overview
Reinforcement Learning assignment with two parts:
- **Part 1 (40 pts)**: Q-Learning and Policy Iteration on Frozen Lake
- **Part 2 (60 pts)**: Deep Q-Learning on an Atari Game

## Environment
- Python with gymnasium, numpy, pandas, seaborn, matplotlib, pytorch
- Notebook: `frozenlake_q_learning.ipynb` (Part 1)
- Part 2 will need a separate notebook

## Plan

### Part 1: Q-Learning & Policy Iteration on Frozen Lake
The notebook already contains the tutorial code (Q-learning agent, epsilon-greedy, training loop, visualization). We need to extend it with:

#### Task 4 — Hyperparameter Analysis
Add cells after the tutorial code that sweep hyperparameters and plot results:
1. **Alpha (learning rate)**: Test 3+ values (e.g., 0.1, 0.5, 0.8) on a fixed map size (e.g., 5x5), hold gamma=0.95, epsilon=0.1 constant. Plot cumulative rewards for each.
2. **Gamma (discount factor)**: Test 3+ values (e.g., 0.5, 0.8, 0.99), hold alpha=0.8, epsilon=0.1. Plot cumulative rewards.
3. **Epsilon (exploration rate)**: Test 3+ values (e.g., 0.01, 0.1, 0.5), hold alpha=0.8, gamma=0.95. Plot cumulative rewards.
4. Add markdown cells explaining the effect of each hyperparameter.

#### Task 5 — Policy Iteration
1. Implement Policy Iteration (policy evaluation + policy improvement loop) for FrozenLake.
2. Extract the transition model P from the environment (`env.P`).
3. Compare the resulting policy and value function to Q-learning's learned Q-table.
4. Add discussion: convergence speed, optimality, model-based vs model-free tradeoffs.

### Part 2: Deep Q-Learning on Atari
Create a new notebook `atari_dqn.ipynb`:

1. **Choose environment**: Pick a manageable Atari game (e.g., `PongNoFrameskip-v4` or `BreakoutNoFrameskip-v4`).
2. **Preprocessing**: Frame stacking, grayscale, resizing (standard Atari wrappers from `gymnasium.wrappers`).
3. **DQN Architecture**: CNN that takes stacked frames as input, outputs Q-values per action.
4. **Replay Buffer**: Experience replay for stable training.
5. **Training Loop**: Epsilon-greedy exploration with decay, target network, periodic updates.
6. **Evaluation**: Plot training rewards over episodes, show gameplay frames, compare to random agent.

## Commands
```bash
# Install dependencies
pip install gymnasium[toy_text,atari,accept-rom-license] torch numpy pandas seaborn matplotlib tqdm ale-py
```
