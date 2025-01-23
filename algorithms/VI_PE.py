import numpy as np
from env.CliffWalk import CliffWalk
from rl_utilities import get_optimal_policy


class VI_PE:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def train(self, num_iteration):
        num_states = self.mdp.num_states()
        r_pi = self.mdp.R()[self.policy, np.arange(num_states)]
        P_pi = self.mdp.P()[self.policy, np.arange(num_states), :]
        self.V_trace = np.zeros((num_iteration, num_states))
        V = np.zeros((num_states))
        for k in range(num_iteration):
            self.V_trace[k, :] = V
            V = r_pi + self.mdp.discount() * P_pi @ V
        

    def run(self, num_iteration, output_filename, save_to_file=True):
        self.train(num_iteration)
        if save_to_file:
            np.save(output_filename, self.V_trace)
        return self.V_trace

