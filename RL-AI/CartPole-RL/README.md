CartPole-v1 DQN Project

A Deep Q-Network (DQN) implementation to solve the CartPole-v1 environment using Reinforcement Learning. This project explores three key approaches: a Baseline DQN, Reward Shaping, and Hyperparameter Tuning to improve the agent's performance.

Project Overview
Environment:

Gymnasium: CartPole-v1
Frameworks: PyTorch, Matplotlib

Objective: Balance a pole on a cart for as long as possible using Reinforcement Learning.
Success Criterion: Achieve an average reward >200 over the last 50 episodes.

1. Methods Implemented

1.1 Baseline DQN:
Implemented a standard DQN agent with default hyperparameters.
Used ε-greedy policy for exploration-exploitation trade-off.
Plotted Total Reward, Loss Over Time, and Epsilon Decay.

Key Observations:
Moderate learning curve.
Reward plateaued below 200 without tuning.
Fluctuations in loss observed.

<img src="https://github.com/user-attachments/assets/a68a26a5-450d-40dc-a421-ca9374d74609" />


