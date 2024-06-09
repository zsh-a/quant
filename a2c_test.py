import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

learning_rate = 1e-3
gamma = 0.99
entropy_beta = 0.01
value_loss_coef = 0.5
max_steps = 200000
update_interval = 5
hidden_size = 128

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value


class A2CAgent:
    def __init__(self, env, model, optimizer, gamma, entropy_beta, value_loss_coef):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.value_loss_coef = value_loss_coef

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        policy_logits, _ = self.model(state)
        policy = torch.softmax(policy_logits, dim=-1)
        action = torch.multinomial(policy, 1).item()
        return action, policy_logits

    def compute_returns(self, rewards, dones, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def update(self, rewards, dones, values, log_probs, entropies, next_value):
        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)

        advantages = returns - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        entropy_loss = entropies.mean()

        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_beta * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = ActorCritic(input_dim, hidden_size, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    agent = A2CAgent(env, model, optimizer, gamma, entropy_beta, value_loss_coef)

    state = env.reset()
    state = state[0]
    episode_rewards = []
    episode_reward = 0

    tot_reward = 0
    for step in range(max_steps):
        log_probs = []
        values = []
        rewards = []
        dones = []
        entropies = []

        for _ in range(update_interval):
            action, policy_logits = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            _, value = model(torch.tensor(state, dtype=torch.float32))
            log_prob = torch.log_softmax(policy_logits, dim=-1)[action]
            entropy = -(torch.softmax(policy_logits, dim=-1) * torch.log_softmax(policy_logits, dim=-1)).sum()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            entropies.append(entropy)

            state = next_state
            episode_reward += reward
            tot_reward += reward
            if done:
                state = env.reset()
                state = state[0]
                tot_reward = 0
                episode_rewards.append(episode_reward)
                episode_reward = 0

        _, next_value = model(torch.tensor(state, dtype=torch.float32))
        agent.update(rewards, dones, values, log_probs, entropies, next_value)

        if len(episode_rewards) % 100 == 0:
            print(f"episode: {len(episode_rewards)}, Total Reward: {tot_reward}")
            # eval_env = gym.make('CartPole-v1',render_mode='human')
            # state = eval_env.reset()[0]
            # done = False
            # total_reward = 0

            # while not done:
            #     action, _ = agent.select_action(state)
            #     state, reward, done,_, _ = eval_env.step(action)
            #     total_reward += reward

            # print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()
