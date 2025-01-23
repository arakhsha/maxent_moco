import numpy as np


class VI_Control:
    def __init__(self, mdp,):
        self.mdp = mdp

    def train(self, num_iteration):
        num_states, num_actions = self.mdp.num_states(), self.mdp.num_actions()
        R = self.mdp.R()
        P = self.mdp.P()
        discount = self.mdp.discount()
        self.V_trace = np.zeros((num_iteration, num_states))
        self.policy_trace = np.zeros((num_iteration, num_states))
        V = np.zeros((num_states))

        for k in range(num_iteration):
            self.V_trace[k, :] = V
            self.policy_trace[k, :] = (R.reshape((num_states * num_actions)) + discount * P.reshape((-1, num_states)) @ V).reshape((-1, num_states)).argmax(axis=0)
            new_V = (R.reshape((num_states * num_actions)) + discount * P.reshape((-1, num_states)) @ V).reshape((-1, num_states)).max(axis=0)
            V = new_V

    def run(self, num_iteration, policy_filename, value_filename, save_to_file=True):
        self.train(num_iteration)
        if save_to_file:
            np.save(value_filename, self.V_trace)
            np.save(policy_filename, self.policy_trace)
        return self.V_trace, self.policy_trace


