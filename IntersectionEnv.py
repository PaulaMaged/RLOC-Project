import numpy as np
from collections import deque
import logging
import math

class IntersectionEnv:
    def __init__(self, arrival_rates, dequeue_rate, change_penalty, max_steps=3600, binDef = [("Empty", 0), ("Medium", 4), ("High", math.inf)]):
        self.arrival_rates = arrival_rates
        self.dequeue_rate = dequeue_rate
        self.change_penalty = change_penalty
        self.max_steps = max_steps
        self.binDef = binDef
        self.reset()

    def reset(self):
        """Resets the environment for a new episode."""
        # Deques now store the exact timestamp a car arrived, NOT its current wait time
        self.queues = [deque() for _ in range(8)] 
        
        # O(1) Running total tracker for cumulative wait time
        self.running_wait_time = [0] * 8
        
        self.current_phase = 0 
        self.current_time_step = 0
        return self.get_discrete_state()
    
    @staticmethod
    def getActionString(action):
        match action:
            case 0: return "N/S Left"
            case 1: return "N/S Through"
            case 2: return "E/W Left"
            case 3: return "E/W Through"
            
    def getQueuesStr(self):
        queuesStr = "\n"
        for idx, queue in enumerate(self.queues):
            queuesStr += f"Lane {idx}: {len(queue)}\n"
        
        return queuesStr
    
    def getStateNdQueueStr(self):
        parts = []
        discretized_state = self.get_discrete_state()
        for idx, binLevel in enumerate(discretized_state[:-1]):
            first = f"L{idx}({self.getLevelName(binLevel):<6})"
            parts.append(f"{first:>8}: {len(self.queues[idx]):<3}")
        
        outputStr = "(" + ", ".join(parts) + ", " f"{self.current_phase}" + ")"
        return outputStr
    
    def getLevelName(self, level):
        for pair in self.binDef:
            name, edge = pair
            if(level <= edge):
                return name
        return "UNKNOWN"
    
    @staticmethod
    def getStateStr(state):
        # Map integers to their string representations
        levels = ["Empty", "Low", "High"]
    
        # Process all elements except the last one
        arr = [levels[val] for val in state[:-1]]
        
        arr.append(IntersectionEnv.getActionString(state[-1]))
        return str(tuple(arr))

    def get_discrete_state(self):
        """Maps lane density to 3 discrete bins: Low, Medium, High."""
        binned_queues = []
        for q in self.queues:
            lane_density = len(q)
            for idx, binDef in enumerate(self.binDef):
                _ , edge = binDef
                if(lane_density <= edge):
                    binned_queues.append(idx)
                    break
            else:
                raise Exception("Shouldn't have undefined bin levels for lane")
                
        return tuple(binned_queues + [self.current_phase])

    def _get_lanes_for_phase(self, phase):
        """Maps an action (phase) to the specific lane indices that get green."""
        phase_map = {
            0: [0, 4], # N/S Left
            1: [1, 5], # N/S Through
            2: [2, 6], # E/W Left
            3: [3, 7]  # E/W Through
        }
        return phase_map.get(phase, [])

    def _advance_time_one_step(self, departures_allowed):
        """Advances the simulation clock by 1 second and handles O(1) wait-time math."""
        self.current_time_step += 1
        
        # 1. Update Running Wait Times (O(1) Magic)
        # Every car in a queue waits 1 more second, increasing the penalty by the queue length
        for i in range(8):
            self.running_wait_time[i] += len(self.queues[i])
            
        # 2. Process Poisson Arrivals
        for i in range(8):
            arrivals = np.random.poisson(self.arrival_rates[i])
            for _ in range(arrivals):
                # Store the EXACT timestamp of arrival
                self.queues[i].append(self.current_time_step)
                
        # 3. Process Departures
        if departures_allowed:
            active_lanes = self._get_lanes_for_phase(self.current_phase)
            for lane in active_lanes:
                # Calculate how many cars can depart this second
                # (Using probability or a fractional accumulation based on dequeue_rate)
                if len(self.queues[lane]) > 0 and np.random.rand() < self.dequeue_rate:
                    # Pop the oldest car
                    arrival_time = self.queues[lane].popleft()
                    
                    # Calculate its total lifetime wait
                    actual_wait_time = self.current_time_step - arrival_time
                    
                    # Remove its accumulated wait time from our running penalty tracker
                    self.running_wait_time[lane] -= actual_wait_time

    def step(self, action):
        """Executes one agent decision in the environment."""
        reward = 0
        done = False
        phase_changed = (action != self.current_phase)
        # 1. Handle Safety Clearance Interval (5 seconds)

        if phase_changed:
            # logging.info("Phase change occured with a penalty of %i; a 5 time-step timelapse will occur before the action takes places!", self.change_penalty)
            reward -= self.change_penalty
            
            # Simulate 5 seconds where NO cars depart
            for _ in range(5): 
                self._advance_time_one_step(departures_allowed=False)
                if self.current_time_step >= self.max_steps:
                    done = True
                    break
            
            self.current_phase = action

        # 2. Normal Time Step Processing (1 second of green light)
        if not done:
            self._advance_time_one_step(departures_allowed=True)
            if self.current_time_step >= self.max_steps:
                done = True
                
        # 3. Calculate Primary Reward (O(1) Time Complexity)
        # Simply sum the 8 integers in our running total. No loops, no deque iterations!
        total_wait_penalty = sum(self.running_wait_time)
        reward -= total_wait_penalty
        # logging.info("Incurred a wait time penalty of %s", total_wait_penalty)
        next_state = self.get_discrete_state()
        
        return next_state, reward, done