
class RoundRobinAgent():
    def __init__(self, action_space_size = 4, time_slice = 30):
        self.action_space_size = action_space_size
        self.current_time_step = 0
        self.time_slice = time_slice
    
    def optimalAction(self, state):
        # get action according to state
        self.current_time_step += 1
        action = None
        if(self.current_time_step % self.time_slice == 0):
            action = (state[-1] + 1) % self.action_space_size
        else:
            action = state[-1]
            
        return action
        