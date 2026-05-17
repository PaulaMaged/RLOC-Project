# %%
print("test kernel")

# %%
#-------------------------------------------------
# Imports
#-------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import pickle
from datetime import datetime 
import importlib

import logging
importlib.reload(logging)

logging.basicConfig(
    filename=f'Logs/.log-{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

import IntersectionEnv
importlib.reload(IntersectionEnv)
from IntersectionEnv import IntersectionEnv

import QLearningAgent
importlib.reload(QLearningAgent)
from QLearningAgent import QLearningAgent

# %%
#-------------------------------------------------
# Initialize your Environment here
#-------------------------------------------------

# --- Simulation Parameters ---
arrival_rates = [0.05, 0.01, 0.12, 0.15, 0.02, 0.02, 0.01, 0.02] 
dequeue_rate = 0.5    
change_penalty = 10.0 
steps_per_episode = 3600 

env = IntersectionEnv(arrival_rates, dequeue_rate, change_penalty, max_steps=steps_per_episode)

# %%
#-------------------------------------------------
# Initialize your agent
#-------------------------------------------------

agent = QLearningAgent(action_space_size=4, alpha=0.1, gamma=0.95, epsilon=0.0)
curr_episode = 0

agentQtable_file_name = "trained_green_wave_model3.pkl"

from pathlib import Path
config_file_path = Path("config.yaml")
if config_file_path.exists():
    import yaml
    with open("config.yaml", 'r') as f:
        try:
            config = yaml.safe_load(f)
            agentQtable_file_name = config["persistence"]["name"]
            print(f"loaded pkl file name: {agentQtable_file_name}")
        except Exception as e:
            print(e)

agent.load_q_table(agentQtable_file_name)

# %%

#-------------------------------------------------
# Start Training Session Here
#-------------------------------------------------

# 1. The Epsilon Schedule
agent.epsilon = 1.0         # Start at 100% exploration
agent.epsilon_min = 0.05    # Never drop below 5% exploration
agent.epsilon_decay = 0.9998 # Slow decay: will hit 5% around episode 15000

episodes = 20000 # Give it slightly more time to fine-tune

episode_rewards = []
avg_reward = 0

logging.disable(level=logging.CRITICAL)  # Disable all logs during training for cleaner output

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
        
        # logging.info("Step %5d| Reward: %-5d | Total Reward: %s)", env.current_time_step, reward, f"{total_reward:,}")
        # logging.info(env.getStateNdQueueStr(ver=1))
        
        state = next_state
        total_reward += reward
        
    episode_rewards.append(total_reward)

    agent.decay_epsilon()
    

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode: {episode + curr_episode + 1:4d} | Epsilon: {agent.epsilon:.3f} | Avg Reward (Last 100): {avg_reward:.0f}")
        
    # if (episode + 1) % 500 == 0:
    #     agent.save_q_table("temp.pkl")
    #     logging.critical("Trained for 500 episodes, 3600 steps each. Avg Reward (Last 500): %.0f", avg_reward)
    #     new_avg_reward = np.mean(episode_rewards[-500:])
    #     if (new_avg_reward / avg_reward) < 5:
    #         logging.critical("New Average Reward is 5 times better than previous!")
    # if (episode + 1) % 500 > 450:
    #     logging.critical("Step %5d| Reward: %-5d | Total Reward: %s)", env.current_time_step, reward, f"{total_reward:,}")
    #     logging.critical(env.getStateNdQueueStr(ver=1))
        
curr_episode += episodes
print("Training complete!")
#%%

#-------------------------------------------------
# Test Current Policy Here
#-------------------------------------------------

logging.disable(level=logging.NOTSET)  

print(f"Starting testing phase for: {type(agent)}...")
logging.critical("Starting testing phase for: %s...", str(type(agent)))

# --- Testing Loop ---
test_episodes = 100
test_rewards = []

for episode in range(test_episodes):

    logging.info(f"{'='*6}Entering new episode - Episode {episode}{'='*6}")
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.optimalAction(state)  # Always choose the best action
        
        stateNdQueueStrBefore = env.getStateNdQueueStr()
        timeStep = env.current_time_step
        
        next_state, reward, done = env.step(action)
        
        state = next_state
        total_reward += reward
        
        logging.info("Step %5d| Reward: %-5d | Total Reward: %s)", env.current_time_step, reward, f"{total_reward:,}")
        logging.info(env.getStateNdQueueStr(ver=1))

        # print(f"Episode: {episode + 1:3d}| State: {state} | Action: {action} | Reward: {reward:.0f} | Total Reward: {total_reward:.0f}")
        
    test_rewards.append(total_reward)
    print(f"Test Episode: {episode + 1:3d} | Total Reward: {total_reward:.0f}")
    
# %%

# Save the trained Q-table to a file
saveToFileName = input("Enter name to save to different file than originaly loaded from: ").strip() or agentQtable_file_name
agent.save_q_table(saveToFileName)
print(len(agent.q_table))

#%%
print(len(agent.q_table))

# %%
