# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_3")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self, gamma=0.99, lambda_=0.95):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 64
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########

        self.relu = nn.ReLU()

        # Shared layers
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.xavier_normal_(self.shared_layer.weight)
        self.shared_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.shared_layer2.weight)
        
        # action layer
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        nn.init.xavier_normal_(self.action_layer.weight)

        #value layer
        self.value_layer = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_normal_(self.value_layer.weight)

        # Initialize GAE for advantage estimation
        self.gae = GAE(gamma, lambda_)

        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.dones = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

        # Pass through shared layers
        x = self.shared_layer(state)
        x = self.relu(x)
        x = self.shared_layer2(x)
        x = self.relu(x)

        # Compute action probabilities
        logits = self.action_layer(x)
        action_prob = F.softmax(logits, dim=-1)

        # Compute state value
        state_value = self.value_layer(x)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        
        action_prob, state_value = self.forward(torch.tensor(state))

        # Create a categorical distribution from the probabilities
        m = Categorical(action_prob)
        action = m.sample()

        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999, lambda_=0.95):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########

        # Calculate rewards-to-go
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # Normalize the returns
        returns = (np.array(returns) - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = torch.tensor(returns, dtype=torch.double)

        values = torch.tensor([sa.value for sa in saved_actions])
        # Calculate the advantages using GAE
        advantages = self.gae(self.rewards, values, self.dones)

        for (log_prob, value), advantage in zip(saved_actions, advantages):
            # Calculate the policy loss with GAE
            policy_losses.append(-log_prob * advantage)
        
        policy_loss = torch.stack(policy_losses).sum()
        # Calculate the value loss
        value_loss = F.smooth_l1_loss(values, returns)
        loss = policy_loss + value_loss

        
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.dones[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, dones):
        """
            Implement Generalized Advantage Estimation (GAE) for your value prediction
            TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
            TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########

        advantages = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            done_mask = 1.0 if dones[t] else 0.0
            # Compute the TD error
            delta = rewards[t] + self.gamma * next_value * (1.0 - done_mask) - values[t]
            # Update the GAE
            gae = delta + self.gamma * self.lambda_ * gae * (1.0 - done_mask)
            advantages.insert(0, gae)
            # Update the next value
            next_value = values[t]
        
        advantages = torch.tensor(advantages, dtype=torch.double)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
        
        ########## END OF YOUR CODE ##########

def train(lr=0.01, gamma=0.999, lambda_=0.95):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy(gamma=gamma, lambda_=lambda_)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100)
    scheduler = Scheduler.PolynomialLR(optimizer=optimizer, total_iters=9999, power=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########

        for step in range(1, 10000):
            # Select an action using the current policy
            action = model.select_action(state)
            # Step the environment
            state, reward, done, _ = env.step(action)
            # Store rewards
            model.rewards.append(reward)
            model.dones.append(done)

            ep_reward += reward
            t += 1

            if done:
                break
            
        optimizer.zero_grad()
        loss = model.calculate_loss(gamma=gamma, lambda_=lambda_)
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########

        writer.add_scalar('Reward/Episode', ep_reward, i_episode)
        writer.add_scalar('EpisodeLength', t, i_episode)
        writer.add_scalar('EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], i_episode)
        writer.add_scalar('Loss', loss, i_episode)  
         
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > 120:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_lr{}_gamma{}_l{}.pth'.format(lr, gamma, lambda_))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = False
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 5e-3
    gamma=0.99
    lambda_=0.8
    env = gym.make('LunarLander-v2')
    env.seed(random_seed) 
    torch.manual_seed(random_seed)  
    train(lr, gamma, lambda_)
    test(f'LunarLander_lr{lr}_gamma{gamma}_l{lambda_}.pth')
