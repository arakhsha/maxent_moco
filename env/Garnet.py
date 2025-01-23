from env.AbstractMDP import AbstractMDP
import numpy as np


class Garnet(AbstractMDP):

    def __init__(self, discount, num_states=10, num_actions=2, b_P=5, b_R=5):
        self._num_states = num_states
        self._num_actions = num_actions
        self.b_P = b_P  # branching factor: number of possible next states for each (s,a) pair
        self.b_R = b_R  # number of non-zero rewards

        self._P = [np.zeros((num_states, num_states)) for _ in range(num_actions)] #List of |A| many |S| x |S| transition matrices
        self._R = np.zeros((num_states, 1)) #np.array of shape |S|x1

        self._populate_P()
        self._populate_R()

        P = np.zeros((num_actions, num_states, num_states))
        R = np.zeros((num_actions, num_states))
        for action in range(num_actions):
            P[action, :, :] = self._P[action]
            R[action, : ] = self._R.reshape(-1)
        
        initial_dist = np.zeros((self._num_states))
        initial_dist[0] = 1
        super().__init__(P, R, discount, initial_dist)


    def get_initial_state_dist(self):
        initial_dist = np.zeros((self._num_states))
        initial_dist[0] = 1
        return initial_dist


    # Setup up the transition probability matrix. Garnet-like (not exact implementation).
    def _populate_P(self):
        for a in range(self._num_actions):
            for s in range(self._num_states):
                p_row = np.zeros(self._num_states)
                indices = np.random.choice(self._num_states, self.b_P, replace=False)
                p_row[indices] = self._generate_stochastic_row(length=self.b_P) #Insert the non-zero transition probabilities in to P
                self._P[a][s, :] = p_row

    def _generate_stochastic_row(self, length):
        p_vec = np.append(np.random.uniform(0, 1, length- 1), [0, 1])
        return np.diff(np.sort(p_vec))  # np.array of length b_P

    #Set up the reward vector
    def _populate_R(self):
        self._R[np.random.choice(self._num_states, self.b_R, replace=False)] = np.random.uniform(0, 1, self.b_R)[:, np.newaxis]


