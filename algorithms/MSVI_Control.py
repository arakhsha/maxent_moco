import numpy as np
from rl_utilities import value_iteration

class MSVI_Control:
    def __init__(self, mdp, Phat):
        self.mdp = mdp
        self.Phat = Phat

    def train(self, num_iteration):
        R, P = self.mdp.R(), self.mdp.P()
        Phat = self.Phat
        num_states, num_actions = self.mdp.num_states(), self.mdp.num_actions()
        self.V_trace = np.zeros((num_iteration, num_states))
        self.policy_trace = np.zeros((num_iteration, num_states))
        V = np.zeros(num_states)
        for k in range(num_iteration):
            self.V_trace[k, :] = V
            r_k = R.reshape((num_states * num_actions)) + self.mdp.discount() * (P - Phat).reshape((-1, num_states)) @ V
            policy = (r_k + self.mdp.discount() * Phat.reshape((-1, self.mdp.num_states())) @ V).reshape((-1, self.mdp.num_states())).argmax(axis=0)
            self.policy_trace[k, :] = policy
            V = value_iteration(Phat, r_k, self.mdp.discount(), err=1e-6, max_iteration=10000)


    def run(self, num_iteration, policy_filename, value_filename, save_to_file=True):
        self.train(num_iteration)
        if save_to_file:
            np.save(value_filename, self.V_trace)
            np.save(policy_filename, self.policy_trace)
        return self.V_trace, self.policy_trace



