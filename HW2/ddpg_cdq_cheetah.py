# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

env_name = 'HalfCheetah-v3'
# Define a tensorboard writer
writer = SummaryWriter(f"./tb_record_{env_name}_cdq")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_outputs)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = self.relu(self.linear1(inputs))
        x = self.relu(self.linear2(x))
        x = self.tanh(self.linear3(x))
        return x
        
        
        ########## END OF YOUR CODE ##########      
        
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Q1
        self.linear1_1 = nn.Linear(num_inputs + num_outputs, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear3_1 = nn.Linear(hidden_size, 1)

        # Q2
        self.linear1_2 = nn.Linear(num_inputs + num_outputs, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3_2 = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        # Q1 forward
        x1 = self.relu(self.linear1_1(xu))
        x1 = self.relu(self.linear2_1(x1))
        x1 = self.linear3_1(x1)

        # Q2 forward
        x2 = self.relu(self.linear1_2(xu))
        x2 = self.relu(self.linear2_2(x2))
        x2 = self.linear3_2(x2)

        return x1, x2

    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.relu(self.linear1_1(xu))
        x1 = self.relu(self.linear2_1(x1))
        x1 = self.linear3_1(x1)
        return x1
    
class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.actor = self.actor
        self.actor_target = self.actor_target
        self.actor_perturbed = self.actor_perturbed
        self.critic = self.critic
        self.critic_target = self.critic_target

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 
        if action_noise is not None:
            mu += torch.tensor(action_noise.noise())
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0]) # Clipping

        return mu



        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))

        noise_std = 0.2
        noise_clip = 0.5
        noise = torch.clamp(torch.randn_like(action_batch) * noise_std, -noise_clip, noise_clip)
        
        # Add noise to target action
        next_action = self.actor_target(next_state_batch) + noise
        # Clip the perturbed action to valid bounds
        next_action = torch.clamp(next_action, self.action_space.low[0], self.action_space.high[0])

        # Get target Q-values from both target critics
        target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
        target_q = torch.min(target_q1, target_q2)

        # Compute target value
        expected_q = reward_batch + self.gamma * target_q * mask_batch

        current_q1, current_q2 = self.critic(state_batch, action_batch)

        value_loss = F.mse_loss(current_q1, expected_q) + F.mse_loss(current_q2, expected_q)

        # Update Critic Network
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        policy_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # === Soft update targets === #
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_cdq_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_cdq_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train():   
    num_episodes = 500
    gamma = 0.99
    tau = 0.02
    lr_a = 1e-5
    lr_c = 5e-3
    hidden_size = 128
    noise_scale = 0.3
    replay_size = 100000
    batch_size = 512
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0
    save_name = f'{env_name}_lra{lr_a}_lrc{lr_c}_g{gamma}_t{tau}_h{hidden_size}_b{batch_size}_n{noise_scale}'
    
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size, lr_a, lr_c)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state = torch.from_numpy(env.reset()).float().unsqueeze(0)
        episode_reward = 0
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            # Select action
            action = agent.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            reward = torch.tensor([[reward]], dtype=torch.float32)
            mask = torch.tensor([[not done]], dtype=torch.float32)

            # Push the sample
            memory.push(state, action, mask, next_state, reward)

            state = next_state
            episode_reward += reward.item()
            total_numsteps += 1

            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    # Update networks
                    actor_loss, critic_loss = agent.update_parameters(batch)

                    writer.add_scalar('loss/value', actor_loss, updates)
                    writer.add_scalar('loss/policy', critic_loss, updates)
                    updates += 1

            if done:
                break


            ########## END OF YOUR CODE ########## 
            # For wandb logging
            # wandb.log({"actor_loss": actor_loss, "critic_loss": critic_loss})

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            # state = torch.Tensor([env.reset()])
            state = torch.from_numpy(env.reset()).float().unsqueeze(0)
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                
                # env.render()
                
                episode_reward += reward

                # next_state = torch.Tensor([next_state])
                next_state = torch.from_numpy(next_state).float().unsqueeze(0)

                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            writer.add_scalar('reward', episode_reward, i_episode)
            writer.add_scalar('ewma_reward', ewma_reward, i_episode)
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))
            
    agent.save_model(save_name+f'_re{ewma_reward}', '.pth')        
 
def test(actor_path, critic_path, num_episodes=10):
    agent = DDPG(env.observation_space.shape[0], env.action_space)
    agent.load_model(actor_path, critic_path)
    for i_episode in range(num_episodes):
        state = torch.from_numpy(env.reset()).float().unsqueeze(0)
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward
            next_state = torch.Tensor([next_state])
            state = next_state
            if done:
                break
        print("Episode: {}, reward: {:.2f}".format(i_episode, episode_reward))

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    env = gym.make(env_name)
    env.seed(random_seed)  
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    train()

    # actor_path = "preTrained\ddpg_cdq_actor_HalfCheetah-v3_lra1e-05_lrc0.005_g0.99_t0.02_h128_b512_n0.3_re5088.744870624083_04182025_225131_.pth"
    # critic_path = "preTrained\ddpg_cdq_critic_HalfCheetah-v3_lra1e-05_lrc0.005_g0.99_t0.02_h128_b512_n0.3_re5088.744870624083_04182025_225131_.pth"
    # test(actor_path, critic_path)