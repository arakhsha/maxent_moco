import numpy as np
from rl_utilities import sample_one_uniform_from_mdp, value_iteration, get_policy_value_mdp
from tqdm import tqdm

class MSDyna_Control:

    def __init__(self, mdp, model):
        self.mdp = mdp
        self.model = model

    def train(self, num_iteration, lr_scheduler):
        num_actions, num_states, discount = self.mdp.num_actions(), self.mdp.num_states(), self.mdp.discount()
        V = np.zeros(num_states)
        r_k = np.zeros((num_actions, num_states))
        self.policy_trace = np.zeros((num_iteration, num_states))
        self.V_trace = np.zeros((num_iteration, num_states))
        self.V_pi_trace = np.zeros((num_iteration, num_states))

        with tqdm(iter(range(num_iteration)), desc="MSDyna", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                self.V_trace[k, :] = V
                P_hat = self.model.get_P_hat()
                policy = (r_k.reshape((num_states * num_actions)) + discount * P_hat.reshape((-1, num_states)) @ V).reshape((-1, num_states)).argmax(axis=0)
                self.policy_trace[k, :] = policy
                self.V_pi_trace[k,:] = get_policy_value_mdp(self.mdp, policy, err=1e-6, max_iteration=100000)
                (s, a, r, next_s) = sample_one_uniform_from_mdp(self.mdp)
                self.model.update(np.array([(s, a, r, next_s)]))
                target = r + discount * V[next_s] - discount * P_hat[a,s,:] @ V
                r_k[a,s] += lr_scheduler.get_lr(current_iter=k) * (target - r_k[a,s])
                V = value_iteration(P_hat, r_k.reshape((num_states * num_actions)), discount, err=1e-6, max_iteration=100000)

    def run(self, num_iteration, lr_scheduler, policy_filename, value_filename, save_to_file=True):
        self.train(num_iteration, lr_scheduler)
        if save_to_file:
            np.save(value_filename, self.V_pi_trace)
            np.save(policy_filename, self.policy_trace)
        return self.V_trace, self.V_pi_trace, self.policy_trace
