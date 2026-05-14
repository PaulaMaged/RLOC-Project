import numpy as np
import pickle
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space_size=4, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        """
        Initializes the Tabular Q-Learning Agent.
        alpha: Learning rate
        gamma: Discount factor (importance of future rewards)
        epsilon: Initial exploration rate
        epsilon_decay: Rate at which exploration shifts to exploitation
        epsilon_min: Minimum exploration rate
        """
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: Maps a state tuple to an array of 4 Q-values (one for each phase)
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

    def choose_action(self, state):
        """Selects an action using the epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Explore: Choose a random phase (0, 1, 2, or 3)
            return np.random.randint(self.action_space_size)
        else:
            # Exploit: Choose the phase with the highest Q-value for the current state
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """Updates the Q-table using the Temporal Difference (TD) target."""
        # Find the maximum expected future reward from the next state
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        
        # Calculate the TD error
        td_error = td_target - self.q_table[state][action]
        
        # Update the Q-value
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Reduces the exploration rate over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def optimalAction(self, state):
        return np.argmax(self.q_table[state])
    
    def save_q_table(self, filename="q_table.pkl"):
        """
        Exports the Q-table to a binary pickle file.
        We cast it to a standard dict to avoid lambda serialization errors.
        """
        try:
            with open(filename, 'wb') as f:
                # Convert defaultdict to a standard dict for safe pickling
                pickle.dump(dict(self.q_table), f)
            print(f"Success: Q-table saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, filename="q_table.pkl"):
        """
        Imports a trained Q-table from a pickle file and reconstructs the defaultdict.
        """
        try:
            with open(filename, 'rb') as f:
                loaded_dict = pickle.load(f)
            
            # Reconstruct the defaultdict using your original lambda factory
            # Make sure 'self.action_space' matches how you define your actions (4 phases)
            self.q_table = defaultdict(lambda: np.zeros(self.action_space))
            
            # Populate it with the loaded data
            self.q_table.update(loaded_dict)
            print(f"Success: Q-table loaded from {filename}. Ready to evaluate!")
            
        except FileNotFoundError:
            print(f"Notice: No existing Q-table found at '{filename}'. Starting with a fresh table.")
        except Exception as e:
            print(f"Error loading Q-table: {e}")
