from pprint import pp
import gymnasium
from gymnasium.spaces.utils import flatten_space
import gym_examples
import logging
from logging import getLogger
logger = getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(state_dim, hidden_dim, action_dim):
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
    )
    
def obs_to_tensor(obs):
    agent = obs["agent"]
    target = obs["target"]
    return torch.tensor([agent[0], agent[1], target[0], target[1]], dtype=torch.float32).to(DEVICE)

@torch.no_grad()
def play_game(env, device, flattened_obs_dim, model, p_random_action):
    obs, _ = env.reset()
    done = False
    max_game_steps = 100
    current_game_step = 0
        
    # start with random state
    states = [
        torch.tensor([0, 0, 0, 0], dtype=torch.float32).to(DEVICE)
    ]
    actions = []
    rewards = []
        
    while current_game_step < max_game_steps and not done:
        current_game_step += 1
            
        if torch.rand(1) < p_random_action:
            action = env.action_space.sample()
        else:
            obs_tensor = obs_to_tensor(obs)                
            action = model(torch.tensor(obs_tensor, dtype=torch.float32).to(device)).argmax().item()
                
        next_obs, reward, done, _, info = env.step(action)
        # logger.debug(f"obs: {obs}, action: {action}, reward: {reward}, done: {done}, info: {info}, game_step: {current_game_step}")
                        
        states.append(obs_to_tensor(next_obs))
        actions.append(torch.tensor(action , dtype=torch.float32))
        rewards.append(torch.tensor(reward, dtype=torch.float32))
            
        obs = next_obs
        env.render()
        
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    rewards = torch.stack(rewards).to(device)
        
    next_states = states[1:].view(-1, flattened_obs_dim)
    states = states[:-1].view(-1, flattened_obs_dim)
    actions = actions.view(-1)
    rewards = rewards.view(-1)
    
    return states, actions, rewards, next_states
        

def run():
    
    env = gymnasium.make('gymnasium_env/Warhammer40k-v0')
    env.reset()
    
    flattened_obs_dim = flatten_space(env.observation_space).shape[0]
    logger.debug(f"flattened_obs_dim: {flattened_obs_dim}")
        
    model = create_model(flattened_obs_dim, 4, env.action_space.n).to(DEVICE)
    
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    p_random_action = 0.1
    
    for i in range(1000):
        states, actions, rewards, next_states =  play_game(env, DEVICE, flattened_obs_dim, model, p_random_action)
        logging.debug(f"mean reward: {rewards.mean()}")
        
        
    gamma = 0.1
    with torch.no_grad():
        q_values = model(next_states).to(DEVICE)
        max_q_values, _ = q_values.max(dim=1)
        expected_future_rewards = rewards + gamma * max_q_values


    # train
    num_epochs = 4
    batch_size = 32
    all_losses = []
    for epoch in range(num_epochs):
        shuffled_indices = torch.randperm(states.size(0))
        states = states[shuffled_indices]
        actions = actions[shuffled_indices]
        expected_future_rewards = expected_future_rewards[shuffled_indices]
        
        for state, action, expected_future_reward in zip(states.split(batch_size), actions.split(batch_size), expected_future_rewards.split(batch_size)):
            logger.debug(f"state: {state}")
            
            q_value = model(state)
            
            logger.debug(f"q_value: {q_value}")
        #     loss = (q_value - expected_future_reward).pow(2)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     all_losses.append(loss.item())
            
        # all_losses = torch.stack(all_losses)
        # logger.debug(f"epoch: {epoch}, loss: {all_losses.mean()}")