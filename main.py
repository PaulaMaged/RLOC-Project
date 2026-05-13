
import numpy as np

from IntersectionEnv import IntersectionEnv
from QLearningAgent import QLearningAgent


# --- Simulation Parameters ---
# Arrival rates (lambda) for the 8 lanes (vehicles per second)
# Assuming a uniform flow baseline for now
arrival_rates = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] 
dequeue_rate = 0.5    # Vehicles cleared per second on green
change_penalty = 10.0 # Eta: Penalty for switching phases to prevent flickering

episodes = 2000
steps_per_episode = 3600 # 1 hour of simulated time per episode

# --- Initialization ---
env = IntersectionEnv(arrival_rates, dequeue_rate, change_penalty)
agent = QLearningAgent(action_space_size=4)

# Tracking metrics for evaluation
episode_rewards = []

print("Starting training phase...")

# --- Main Training Loop ---
for episode in range(episodes):
    # Reset the environment at the start of each episode
    env.__init__(arrival_rates, dequeue_rate, change_penalty)
    state = env.get_discrete_state()
    total_reward = 0
    
    for step in range(steps_per_episode):
        # 1. Agent chooses an action (phase)
        action = agent.choose_action(state)
        
        # 2. Environment steps forward based on the action
        next_state, reward = env.step(action)
        
        # 3. Agent learns from the consequences
        agent.learn(state, action, reward, next_state)
        
        # 4. Transition to the next state
        state = next_state
        total_reward += reward
        
    # Decay exploration rate at the end of the episode
    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    
    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode: {episode + 1:4d} | Epsilon: {agent.epsilon:.3f} | Avg Reward (Last 100): {avg_reward:.0f}")

print("Training complete!")