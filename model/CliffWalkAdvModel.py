import numpy as np
from env.CliffWalk import CliffWalk
from model.AbstractModel import AbstractModel


class CliffWalkAdvModel(AbstractModel):

    def __init__(self, num_states, num_actions, alpha, adv_mdp_success_prob=0.9):
        self.num_states = num_states
        self.num_actions = num_actions

        if num_states != 36 or num_actions != 4:
            assert False

        self.counts = np.zeros((num_actions, num_states, num_states))
        self.rhat = np.zeros((num_actions, num_states))
        self.alpha = alpha

        Cliffwalk_P = CliffWalk(adv_mdp_success_prob, 0.99).P()
        self.AdvP = np.zeros((self.num_actions, self.num_states, self.num_states))
        for a in range(self.num_actions):
            self.AdvP[a, :, :] = Cliffwalk_P[CliffWalk.ACTION_UP, :, :]
        super().__init__()


    def update(self, sars_vector):
        for i in range(sars_vector.shape[0]):
            s = int(sars_vector[i, 0])
            a = int(sars_vector[i, 1])
            r = int(sars_vector[i, 2])
            sp = int(sars_vector[i, 3])

            self.counts[a, s, sp] += 1
            new_sa_count = np.sum(self.counts[a, s, :])
            self.rhat[a, s] += 1/new_sa_count * (r - self.rhat[a, s])

    def sample_next_state(self, state, action):
        dist = self.get_P_hat()[action, state, :]
        s = np.random.choice(np.arange(self.num_states), p=dist)
        return s

    def sample_reward(self, state, action):
        # Not implemented
        return None

    def get_P_hat(self):

        totals = self.counts.sum(axis=2, keepdims=True)
        true_count = np.ones((self.num_actions, self.num_states, self.num_states)) / self.num_states
        np.divide(self.counts, totals, out=true_count, where=totals != 0)

        P_hat = true_count * (1 - self.alpha) + self.AdvP * self.alpha
        return P_hat

    def get_r_hat(self):
        return self.rhat        


    @staticmethod
    def get_P_hat_using_P(P, alpha, adv_mdp_success_prob=0.9):
        num_states = P.shape[2]
        num_actions = P.shape[0]

        Cliffwalk_P = CliffWalk(adv_mdp_success_prob, 0.99).P()
        AdvP = np.zeros((num_actions, num_states, num_states))
        for a in range(num_actions):
            AdvP[a, :, :] = Cliffwalk_P[CliffWalk.ACTION_UP, :, :]

        P_hat = P * (1 - alpha) + AdvP * alpha
        return P_hat

