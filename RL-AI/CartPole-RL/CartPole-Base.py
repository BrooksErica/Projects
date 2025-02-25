#import libraries
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy 

# 1. Create the Environment
env = gym.make("CartPole-v1")

# 2. Instantiate the Agent (PPO)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the Agent
model.learn(total_timesteps=50_000)

# 4. Evaluate the Agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Baseline - Mean reward: {mean_reward} +/- {std_reward}")