import torch
import torch.nn as nn
import torch.optim as optim
import gym

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.policy(x)

def ppo_update(policy, optimizer, states, actions, log_probs, returns, advantages, epsilon=0.2, clip_value=0.1):
    # Calculate advantages
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    with torch.autograd.set_detect_anomaly(True):
        for _ in range(3):  # Number of epochs
            for state, action, old_log_prob, return_, advantage in zip(states, actions, log_probs, returns, advantages):
                new_log_prob = torch.log(policy(state.squeeze(0))[action])
                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # print(actor_loss.cpu().item())
                
                optimizer.zero_grad()
                actor_loss.backward()
                
                optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = Policy(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    num_episodes = 1000
    max_steps = 500
    gamma = 0.99

    for episode in range(num_episodes):
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        tot_reward = 0
        state = env.reset()
        state = state[0]
        for step in range(max_steps):
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy(state)
            action = torch.multinomial(action_probs, num_samples=1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action])

            next_state, reward, done, _,_ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            tot_reward += reward

            state = next_state

            if done:
                break

        returns = []
        advantages = []
        R = 0

        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        for i in range(len(rewards)):
            if i == len(rewards) - 1:  # 如果是最后一步
                advantages.append(returns[i] - returns[i])
            else:
                advantages.append(returns[i] - sum(returns[i+1:]) / len(rewards[i+1:]))
        ppo_update(policy, optimizer, states, actions, log_probs, returns, advantages)

        if episode % 100 == 0:
            print("Episode {}: Total Reward = {}".format(episode, tot_reward))

    env.close()

if __name__ == "__main__":
    main()
