# %% [markdown]
# # RL Project - Q-Learning Based Approach to Implement Dynamic Traffic Signals
# The project works on a traffic signal agent that allows for responding to varying states of congestion.

# %%
print("test kernel")

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

episodes = 5000
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
agent.load_q_table("trained_green_wave_model1.pkl")

# %%
agent.epsilon = 0.9
agent.epsilon_decay = 1


episode_rewards = []

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
        
        logging.info(
            "Experience Tuple: (%s, %s, %i, %s)", 
            IntersectionEnv.getStateStr(state), 
            IntersectionEnv.getActionString(action), 
            reward
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


#%%

logging.disable(level=logging.NOTSET)  

print("Starting testing phase...")
# --- Testing Loop ---
test_episodes = 100
test_rewards = []
agent.epsilon = 0.0  # No exploration during testing
for episode in range(test_episodes):

    logging.info("Entering episode: %s", episode)
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.optimalAction(state)  # Always choose the best action
        next_state, reward, done = env.step(action)
        
        logging.info(
            "Experience Tuple: (%s, %s, %i)", 
            IntersectionEnv.getStateStr(state), 
            IntersectionEnv.getActionString(action), 
            reward
        )

        state = next_state
        total_reward += reward

        print(f"Episode: {episode + 1:3d}| State: {state} | Action: {action} | Reward: {reward:.0f} | Total Reward: {total_reward:.0f}")
        
    test_rewards.append(total_reward)
    print(f"Test Episode: {episode + 1:3d} | Total Reward: {total_reward:.0f}")

#%%

# %%

# Save the trained Q-table to a file
agent.save_q_table("trained_green_wave_model1.pkl")


print(len(agent.q_table))



# %%
