import numpy as np
from collections import deque

class IntersectionEnv:
    def __init__(self, arrival_rates, dequeue_rate, change_penalty):
        self.arrival_rates = arrival_rates
        self.dequeue_rate = dequeue_rate
        self.change_penalty = change_penalty
        
        # Initialize an array of 8 deques. 
        # Each deque represents a lane. Elements inside are the wait times of individual cars.
        self.queues = [deque() for _ in range(8)] 
        
        self.current_phase = 0 
        self.time_step = 1.0 

    def get_discrete_state(self):
        """
        Maps the density of the lanes to the 3 discrete bins: Low, Medium, High.
        The state is still based on the NUMBER of vehicles (length of the deque).
        """
        binned_queues = []
        for q in self.queues:
            lane_density = len(q)
            if lane_density <= 2:
                binned_queues.append(0) # Low
            elif lane_density <= 6:
                binned_queues.append(1) # Medium
            else:
                binned_queues.append(2) # High
                
        return tuple(binned_queues + [self.current_phase])

    def _process_arrivals(self):
        """Simulates arrivals and ages the cars currently waiting."""
        for i in range(8):
            # 1. Increment the wait time (age) of all cars currently in the queue
            for j in range(len(self.queues[i])):
                self.queues[i][j] += 1
            
            # 2. Process new arrivals via Poisson distribution
            arrivals = np.random.poisson(self.arrival_rates[i] * self.time_step)
            for _ in range(arrivals):
                # New cars start with a wait time of 0
                self.queues[i].append(0)

    def _process_departures(self, active_lanes):
        """Removes the oldest vehicles from lanes with a green light."""
        for i in active_lanes:
            departures = int(np.floor(self.dequeue_rate * self.time_step))
            
            # Pop cars from the left (front of the line) up to the departure limit
            # Ensure we don't try to pop from an empty queue
            for _ in range(min(len(self.queues[i]), departures)):
                self.queues[i].popleft()

    def step(self, action):
        """Executes one time step in the environment."""
        reward = 0
        phase_changed = (action != self.current_phase)
        
        # 1. Handle Safety Clearance Interval (5 seconds)
        if phase_changed:
            for _ in range(5): 
                self._process_arrivals()
                # Accumulate the total wait time penalty during the clearance interval
                reward -= sum(sum(q) for q in self.queues)
            
            reward -= self.change_penalty
            self.current_phase = action

        # 2. Normal Time Step Processing
        self._process_arrivals()
        active_lanes = self._get_lanes_for_phase(self.current_phase) 
        self._process_departures(active_lanes)
        
        # 3. Calculate Primary Reward 
        # Summing the values inside all deques gives us the cumulative wait time of every car
        total_wait_time = sum(sum(q) for q in self.queues)
        reward -= total_wait_time
        
        next_state = self.get_discrete_state()
        
        return next_state, reward

    def _get_lanes_for_phase(self, phase):
        """Maps an action (phase) to the specific lane indices that get green."""
        phase_map = {
            0: [0, 4], # N/S Left
            1: [1, 5], # N/S Through
            2: [2, 6], # E/W Left
            3: [3, 7]  # E/W Through
        }
        return phase_map.get(phase, [])