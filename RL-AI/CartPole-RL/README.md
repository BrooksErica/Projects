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

<img width="280" alt="image" src="https://github.com/user-attachments/assets/a68a26a5-450d-40dc-a421-ca9374d74609" />

1.2 Reward Shaping:

Modified the reward function by adding penalties for:
  Pole angle deviations.
  Cart moving away from the center.
Encouraged smoother balancing behavior.

Key Observations:
Faster convergence.
Smoother reward curve.
Improved average reward compared to baseline.

<img width="280" alt="image" src="https://github.com/user-attachments/assets/36b92d55-d414-4db7-97cf-bbddb54e3aed" />

1.3 Hyperparameter Tuning:

Applied Grid Search across key hyperparameters:
  Learning Rate: 0.0005, 0.001
  Gamma (Discount Factor): 0.95, 0.99
  Epsilon Decay: 0.995, 0.999
Selected the best configuration based on the highest average reward over the last 50 episodes.

Key Observations:
Improved reward stability.
Higher convergence rate.
Reached success criterion (>200) in most trials.

<img width="280" alt="image" src="https://github.com/user-attachments/assets/180ba6f4-0b1e-4d6b-9cff-0f7c1a50a10e" />







