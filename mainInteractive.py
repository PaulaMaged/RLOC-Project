# %% [markdown]
# # RL Project - Q-Learning Based Approach to Implement Dynamic Traffic Signals
# The project works on a traffic signal agent that allows for responding to varying states of congestion.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import pickle
import logging
from datetime import datetime 

from IntersectionEnv import IntersectionEnv
from QLearningAgent import QLearningAgent

# %%
# --- Simulation Parameters ---
arrival_rates = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] 
dequeue_rate = 0.5    
change_penalty = 10.0 

episodes = 2000
steps_per_episode = 3600 

# --- Initialization ---
env = IntersectionEnv(arrival_rates, dequeue_rate, change_penalty, max_steps=steps_per_episode)
agent = QLearningAgent(action_space_size=4)

curr_episode = 0

# %%
logging.basicConfig(
    filename=f'Logs/app.log-{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# %%
# Initialize your agent
agent = QLearningAgent(action_space_size=4, alpha=0.1, gamma=0.95, epsilon=0.0)

# Load the brain before interactinge with the environment
agent.load_q_table("trained_green_wave_model.pkl")

# %%
agent.epsilon = 0.9

episode_rewards = []

print("Starting training phase...")

# --- Main Training Loop ---
for episode in range(episodes):
    # Use the proper reset method instead of __init__
    
    logging.info("Entering episode: %s", episode)
    state = env.reset()
    total_reward = 0
    done = False
    
    # Use a while loop based on the 'done' flag
    while not done:
        action = agent.choose_action(state)
        
        # Unpack the new 'done' variable from the environment
        next_state, reward, done = env.step(action)
        
        agent.learn(state, action, reward, next_state)
        
        logging.info(
            "Experience Tuple: (%s, %s, %i, %s)", 
            IntersectionEnv.getStateStr(state), 
            IntersectionEnv.getActionString(action), 
            reward, 
            IntersectionEnv.getStateStr(next_state)
        )
        
        state = next_state
        total_reward += reward
        
    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode: {episode + curr_episode + 1:4d} | Epsilon: {agent.epsilon:.3f} | Avg Reward (Last 100): {avg_reward:.0f}")


curr_episode += episodes
print("Training complete!")

# %%

# Save the trained Q-table to a file
agent.save_q_table("trained_green_wave_model1.pkl")

# %%
print("test kernel")

# %%
print(len(agent.q_table))


