import numpy as np

class IntersectionEnv:
    def __init__(self, arrival_rates, dequeue_rate, change_penalty):
        # arrival_rates: list or array of 8 lambda values for Poisson arrivals
        # dequeue_rate: vehicles departing per second on a green light
        # change_penalty: eta (penalty for switching phases)
        
        self.arrival_rates = arrival_rates
        self.dequeue_rate = dequeue_rate
        self.change_penalty = change_penalty
        
        # Track exact vehicle counts for the 8 lanes
        # Indices 0-7 represent the specific directional lanes (e.g., N-Left, N-Through, etc.)
        self.queues = np.zeros(8, dtype=int) 
        
        # Start at Phase 0 (e.g., N/S Left)
        self.current_phase = 0 
        self.time_step = 1.0 # 1 second per simulation step

    def get_discrete_state(self):
        """
        Maps exact vehicle counts to the 3 discrete bins: Low, Medium, High.
        Returns a tuple representing the current state for Q-table lookup.
        """
        binned_queues = []
        for q in self.queues:
            if q <= 2:
                binned_queues.append(0) # Low
            elif q <= 6:
                binned_queues.append(1) # Medium
            else:
                binned_queues.append(2) # High
                
        # State is the 8 binned queues + current phase
        return tuple(binned_queues + [self.current_phase])

    def _process_arrivals(self):
        """Simulates vehicle arrivals using a Poisson process for each lane."""
        for i in range(8):
            # Poisson arrival: probability of arrival based on lambda
            arrivals = np.random.poisson(self.arrival_rates[i] * self.time_step)
            self.queues[i] += arrivals

    def _process_departures(self, active_lanes):
        """Removes vehicles from lanes that currently have a green light."""
        for i in active_lanes:
            if self.queues[i] > 0:
                # Dequeue vehicles, ensuring queue doesn't drop below 0
                departures = int(np.floor(self.dequeue_rate * self.time_step))
                self.queues[i] = max(0, self.queues[i] - departures)

    def step(self, action):
        """
        Executes one time step in the environment.
        action: The chosen phase (0, 1, 2, or 3)
        """
        reward = 0
        phase_changed = (action != self.current_phase)
        
        # 1. Handle Safety Clearance Interval (5 seconds)
        if phase_changed:
            # During transition, no one dequeues, but arrivals still happen
            for _ in range(5): 
                self._process_arrivals()
                # Accumulate wait time penalty during the red/yellow lights
                reward -= np.sum(self.queues) 
            
            # Apply the change penalty (eta)
            reward -= self.change_penalty
            self.current_phase = action

        # 2. Process Arrivals and Departures for the normal time step
        self._process_arrivals()
        
        # Map the current phase to the specific lanes that get a green light
        # (You will need to map your 8 lanes to the 4 phases here)
        active_lanes = self._get_lanes_for_phase(self.current_phase) 
        self._process_departures(active_lanes)
        
        # 3. Calculate primary reward (Negative sum of all waiting vehicles)
        reward -= np.sum(self.queues)
        
        next_state = self.get_discrete_state()
        
        return next_state, reward

    def _get_lanes_for_phase(self, phase):
        """Maps an action (phase) to the specific lane indices that get green."""
        # TODO: Define which of the 0-7 indices correspond to N/S Left, N/S Through, etc.
        phase_map = {
            0: [0, 4], # Example: Lane 0 and 4 are N/S Left
            1: [1, 5], # Example: Lane 1 and 5 are N/S Through
            2: [2, 6], # Example: E/W Left
            3: [3, 7]  # Example: E/W Through
        }
        return phase_map.get(phase, [])