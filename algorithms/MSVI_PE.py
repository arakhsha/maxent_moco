import numpy as np
from rl_utilities import get_policy_value


class MSVI_PE:
    def __init__(self, mdp, policy, Phat):
        self.mdp = mdp
        self.policy = policy
        self.Phat = Phat

    def train(self, num_iteration):
        num_states = self.mdp.num_states()
        num_actions = self.mdp.num_actions()
        R = self.mdp.R()
        P = self.mdp.P()
        Phat = self.Phat
        self.V_trace = np.zeros((num_iteration, num_states))
        V = np.zeros((num_states))
        for k in range(num_iteration):
            self.V_trace[k, :] = V
            rbar = np.zeros((num_actions, num_states))
            for a in range(num_actions):
                rbar[a, :] = R[a, :] + self.mdp.discount() * (P[a, :, :] - Phat[a, :, :]) @ V
            V = get_policy_value(Phat, rbar, self.mdp.discount(), self.policy, err=1e-10)
        

    def run(self, num_iteration, output_filename, save_to_file=True):
        self.train(num_iteration)
        if save_to_file:
            np.save(output_filename, self.V_trace)
        return self.V_trace


