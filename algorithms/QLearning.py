import numpy as np

from rl_utilities import sample_one_uniform_from_mdp, get_policy_value_mdp
from tqdm import tqdm

class QLearning:

    def __init__(self, mdp):
        self.mdp = mdp

    def train(self, num_iteration, lr_scheduler):
        discount = self.mdp.discount()
        num_actions = self.mdp.num_actions()
        num_states = self.mdp.num_states()
        Q = np.zeros((num_actions, num_states))
        self.V_trace = np.zeros((num_iteration, num_states))
        self.V_pi_trace = np.zeros((num_iteration, num_states))
        self.policy_trace = np.zeros((num_iteration, num_states))
        with tqdm(iter(range(num_iteration)), desc="Q-Learning", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                greedy_policy = Q.argmax(axis=0)
                self.policy_trace[k, :] = greedy_policy
                self.V_pi_trace[k,:] = get_policy_value_mdp(self.mdp, greedy_policy, err=1e-6, max_iteration=100000)
                self.V_trace[k, :] = Q[greedy_policy, np.arange(num_states)]
                (s,a,r,next_s) = sample_one_uniform_from_mdp(self.mdp)
                next_a = Q.argmax(axis=0)[next_s]
                target = r + discount * Q[next_a, next_s]
                Q[a, s] = Q[a, s] + lr_scheduler.get_lr(current_iter=k) * (target - Q[a, s])


    def run(self, num_iteration, lr_scheduler, policy_filename, value_filename):
        self.train(num_iteration, lr_scheduler)
        np.save(value_filename, self.V_pi_trace)
        np.save(policy_filename, self.policy_trace)
