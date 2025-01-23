import numpy as np
from rl_utilities import sample_one_from_mdp, get_policy_value
from tqdm import tqdm

class MSDyna_PE:

    def __init__(self, mdp, policy, model):
        self.mdp = mdp
        self.policy = policy
        self.model = model

    def train(self, num_iteration, lr_scheduler):
        discount = self.mdp.discount()
        num_actions, num_states = self.mdp.num_actions(), self.mdp.num_states()
        V = np.zeros(num_states)
        r_k = np.zeros((num_actions, num_states))
        self.V_trace = np.zeros((num_iteration, num_states))
        with tqdm(iter(range(num_iteration)), desc="MSDyna", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                (s,a,r,next_s) = sample_one_from_mdp(self.mdp, self.policy)
                self.model.update(np.array([(s,a,r,next_s)]))
                P_hat = self.model.get_P_hat()
                r_k[a,s] += lr_scheduler.get_lr(current_iter=k) * (r + discount * V[next_s] - discount * P_hat[a,s,:] @ V - r_k[a,s])
                V = get_policy_value(P_hat, r_k, discount, self.policy)
                self.V_trace[k,:] = V

    def run(self, num_iteration, lr_scheduler, output_filename):
        self.train(num_iteration, lr_scheduler)
        np.save(output_filename, self.V_trace)

