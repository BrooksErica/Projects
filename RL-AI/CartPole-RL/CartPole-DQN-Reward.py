
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 500
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000

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

# Reward Shaping Function
def shaped_reward(state, reward):
    # Example: Encourage pole to stay upright and cart to stay near center
    pole_angle = abs(state[2])  # Pole angle
    cart_position = abs(state[0])  # Cart position

    angle_penalty = pole_angle * 0.5
    position_penalty = cart_position * 0.1

    return reward - (angle_penalty + position_penalty)

# Training the DQN Agent
def train():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON
    rewards = []

    for episode in range(EPISODES):
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
            reward = shaped_reward(next_state, reward)  # Apply reward shaping
            next_state = torch.FloatTensor(next_state)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards_batch + (GAMMA * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards.append(total_reward)

        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()

    # Plotting rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN with Reward Shaping on CartPole-v1')
    plt.show()

if __name__ == '__main__':
    train()
