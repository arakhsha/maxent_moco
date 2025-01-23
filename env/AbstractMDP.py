import numpy as np

class AbstractMDP:

    def __init__(self, P, R, discount, initial_dist):
        self._P = P
        self._R = R
        self._initial_dist = initial_dist
        self._discount = discount

        self._num_actions = R.shape[0]
        self._num_states = R.shape[1]


    def P(self):
        return self._P

    def R(self):
        return self._R

    def discount(self):
        return self._discount

    def num_states(self):
        return self._num_states

    def num_actions(self):
        return self._num_actions

    def initial_dist(self):
        return self._initial_dist

    def sample_step(self, state, action):
        transition_vector = self._P[action, state, :]
        next_state = np.random.choice(np.arange(0, self.num_states()), p=transition_vector)
        reward = self._R[action, state]
        return next_state, reward
