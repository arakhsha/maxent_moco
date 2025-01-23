import random
import sys
import time
import numpy as np
# setting path
sys.path.append('../splitting_tabular')
from LearningRate import LearningRate
from algorithms.MECVI_PE import MECVI_PE
from rl_utilities import sample_one_from_mdp, get_policy_value
from tqdm import tqdm

class MECDyna_PE:

    def __init__(self, order, mdp, policy, model):
        self.mdp = mdp
        self.policy = policy
        self.model = model
        self.order = order

    def train(self, num_iteration, num_unused_V, beta, planning_interval, val_queue_interval):
        num_actions, num_states, discount = self.mdp.num_actions(), self.mdp.num_states(), self.mdp.discount()

        R = self.mdp.R()

        V = np.zeros(num_states)
        M = np.zeros((num_states, self.order))
        F = np.zeros((num_actions, num_states, self.order))

        # F approximates R + discount * PM

        self.q_error_trace = np.zeros((num_iteration, self.order))
        self.V_trace = np.zeros((num_iteration, num_states))
        zero_dual_params = np.zeros((num_states, self.order))

        V_queue = []
        PpiV_queue = []
        count_queue = []
        basis_norm = 1000

        for i in range(self.order + num_unused_V):
            new_U = np.random.uniform(size = num_states)

            if len(V_queue) > 0:
                U = (np.array(V_queue)).T / basis_norm
                new_U = new_U - U @ U.T @ new_U
            new_U = basis_norm * new_U / (np.linalg.norm(new_U, ord=2) + 1e-7)
            
            V_queue.append(new_U)
            PpiV_queue.append(np.zeros((num_states)))
            count_queue.append(np.zeros((num_states)))
        
            

        with tqdm(iter(range(num_iteration)), desc=f"MECDyna{self.order}", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                self.V_trace[k, :] = V
                (s, a, r, next_s) = sample_one_from_mdp(self.mdp, self.policy)
                self.model.update(np.array([(s, a, r, next_s)]))

                for d in range(len(V_queue)):
                        count_queue[d][s] += 1
                        lr = 1 / count_queue[d][s]
                        PpiV_queue[d][s] += lr * (V_queue[d][next_s] - PpiV_queue[d][s])

                Phat = self.model.get_P_hat()
                if k == 0:
                    Pbar = np.copy(Phat)
                if k % planning_interval == 0:
                    # if k >= num_unused_V * val_queue_interval:
                    if len(V_queue) > num_unused_V:
                        M = (np.array(V_queue)[:-num_unused_V, :]).T
                        PM = (np.array(PpiV_queue)[:-num_unused_V, :]).T
                        
                        Pbar, _ = MECVI_PE.calc_Pbar_PE(M, PM, Phat, self.policy, beta, zero_dual_params)
                        
                        V = get_policy_value(Pbar, R, self.mdp.discount(), self.policy, err=1e-10)
                    else:
                        V = get_policy_value(Phat, R, self.mdp.discount(), self.policy, err=1e-10)

                if k % val_queue_interval == 0:
                    # print(np.linalg.norm(V, ord=2))
                    new_U = np.copy(V)
                    if len(V_queue) > 0 and self.order > 1:
                        U = (np.array(V_queue)[-self.order+1:]).T / basis_norm
                        new_U = new_U - U @ U.T @ new_U
                    new_U = basis_norm * new_U / (np.linalg.norm(new_U, ord=2) + 1e-7)
                    V_queue.append(new_U)
                    # PV_queue.append((self.mdp.P().reshape((-1, num_states)) @ new_U).reshape((num_actions, num_states)))
                    if len(PpiV_queue) > 0:
                        PpiV_queue.append(np.copy(PpiV_queue[-1]))
                    else:
                        PpiV_queue.append(np.zeros((num_actions, num_states)))
                    count_queue.append(np.zeros((num_states)))
                    if len(V_queue) > self.order + num_unused_V:
                        V_queue.pop(0)
                        PpiV_queue.pop(0)
                        count_queue.pop(0)

                self.V_trace[k, :] = V

    def run(self, num_iteration, num_unused_V, beta, planning_interval, val_queue_interval, output_filename, save_to_file=True):
        self.train(num_iteration, num_unused_V, beta, planning_interval, val_queue_interval)
        if save_to_file:
            np.save(output_filename, self.V_trace)
        return self.V_trace


