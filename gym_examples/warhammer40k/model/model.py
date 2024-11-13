from pprint import pp
import gymnasium
from gymnasium.spaces.utils import flatten_space
import gym_examples
import logging
from logging import getLogger
logger = getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from tqdm import tqdm
import plotly.express as px


import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

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
    
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    
    logging.debug("Playing games")
    for i in range(1000):
        states, actions, rewards, next_states =  play_game(env, DEVICE, flattened_obs_dim, model, p_random_action)
        
        # append to all 
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_next_states.extend(next_states)
    
    all_states = torch.stack(all_states)
    all_actions = torch.stack(all_actions)
    all_rewards = torch.stack(all_rewards)
    all_next_states = torch.stack(all_next_states)
        
    logging.debug(f"mean reward: {all_rewards.mean()}")
        
        
    gamma = 0.1
    with torch.no_grad():
        q_values = model(all_next_states).to(DEVICE)
        max_q_values, _ = q_values.max(dim=1)
        expected_future_rewards = all_rewards + gamma * max_q_values
        

    # train
    num_epochs = 10000
    batch_size = 64
    all_losses = []
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
                
        shuffled_indices = torch.randperm(states.shape[0])
        states = all_states[shuffled_indices]
        actions = all_actions[shuffled_indices]
        expected_future_rewards = expected_future_rewards[shuffled_indices]

        for state, action, expected_future_reward in zip(
            states.split(batch_size),
            actions.split(batch_size),
            expected_future_rewards.split(batch_size)
            ):
            
            q_value = model(state)
                        
            reward_index = action.view(-1, 1).long()
                        
            predicted_future_rewards = q_value.gather(1, reward_index).view(-1)
            loss = mse_loss(predicted_future_rewards, expected_future_reward)
            all_losses.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        pbar.set_description(f"Epoch {epoch} mean loss: {torch.stack(all_losses).mean()}")
    
    data = torch.stack(all_losses).detach().cpu().numpy()
    fig = px.line(data).update_layout(xaxis_title="Epoch", yaxis_title="MSE Loss")
    fig.show()