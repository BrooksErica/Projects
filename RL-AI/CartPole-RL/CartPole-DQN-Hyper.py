
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from itertools import product

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Training Function
def train_dqn(env, hyperparams):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=hyperparams['learning_rate'])
    memory = ReplayBuffer(hyperparams['memory_size'])

    epsilon = hyperparams['epsilon_start']
    rewards = []
    losses = []
    epsilons = []

    for episode in range(hyperparams['episodes']):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        for t in range(500):
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > hyperparams['batch_size']:
                batch = memory.sample(hyperparams['batch_size'])
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards_batch + (hyperparams['gamma'] * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            if done:
                break

        rewards.append(total_reward)
        epsilons.append(epsilon)

        if epsilon > hyperparams['epsilon_min']:
            epsilon *= hyperparams['epsilon_decay']

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    return rewards, losses, epsilons

# Grid Search for Hyperparameters
def grid_search(env, param_grid):
    best_score = -float('inf')
    best_params = None
    results = {}

    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        hyperparams = dict(zip(keys, combo))
        print(f"\nTesting Hyperparameters: {hyperparams}")

        rewards, losses, epsilons = train_dqn(env, hyperparams)
        avg_reward = np.mean(rewards[-50:])  # Last 50 episodes

        results[str(hyperparams)] = avg_reward

        if avg_reward > best_score:
            best_score = avg_reward
            best_params = hyperparams

    return best_params, results

# Plotting Performance Metrics
def plot_metrics(rewards, losses, epsilons):
    plt.figure(figsize=(15, 5))

    # Total Rewards
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title('Loss Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    # Epsilon Decay
    plt.subplot(1, 3, 3)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.show()

# Main Function
def main():
    env = gym.make('CartPole-v1')

    param_grid = {
        'learning_rate': [0.0005, 0.001],
        'gamma': [0.95, 0.99],
        'epsilon_start': [1.0],
        'epsilon_decay': [0.995, 0.999],
        'epsilon_min': [0.01],
        'batch_size': [64],
        'memory_size': [10000],
        'episodes': [300]
    }

    best_params, results = grid_search(env, param_grid)
    print(f"\nBest Hyperparameters: {best_params}")

    # Train with best parameters
    rewards, losses, epsilons = train_dqn(env, best_params)

    # Plot Performance
    plot_metrics(rewards, losses, epsilons)

    env.close()

if __name__ == '__main__':
    main()