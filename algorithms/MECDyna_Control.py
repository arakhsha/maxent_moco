import time
import numpy as np
from algorithms.MECVI_Control import MECVI_Control
from rl_utilities import sample_one_uniform_from_mdp, value_iteration, get_policy_value_mdp
from tqdm import tqdm

class MECDyna_Control:

    def __init__(self, order, mdp, model):
        self.mdp = mdp
        self.model = model
        self.order = order

    def train(self, num_iteration, num_unused_V, beta, planning_interval, val_queue_interval):
        num_actions, num_states, discount = self.mdp.num_actions(), self.mdp.num_states(), self.mdp.discount()

        R = self.mdp.R()

        V = np.zeros(num_states)
        policy = np.zeros(num_states)
        M = np.zeros((num_states, self.order))
        F = np.zeros((num_actions, num_states, self.order))

        # F approximates R + discount * PM

        self.q_error_trace = np.zeros((num_iteration, self.order))
        self.policy_trace = np.zeros((num_iteration, num_states))
        self.V_trace = np.zeros((num_iteration, num_states))
        self.V_pi_trace = np.zeros((num_iteration, num_states))
        self.V_pi_trace = np.zeros((num_iteration, num_states))
        self.Phat_trace = []
        self.Pbar_trace = []
        self.opt_time_trace = []
        self.model_iter_trace = []
        zero_dual_params = np.zeros((num_actions, num_states, self.order))

        V_queue = []
        PV_queue = []
        count_queue = []
        basis_norm = 1000

        for i in range(self.order + num_unused_V):
            new_U = np.random.uniform(size = num_states)

            if len(V_queue) > 0:
                U = (np.array(V_queue)).T / basis_norm
                new_U = new_U - U @ U.T @ new_U
            new_U = basis_norm * new_U / (np.linalg.norm(new_U, ord=2) + 1e-7)
            
            V_queue.append(new_U)
            PV_queue.append(np.zeros((num_actions, num_states)))
            count_queue.append(np.zeros((num_actions, num_states)))
        
            

        with tqdm(iter(range(num_iteration)), desc=f"MECDyna{self.order}", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                self.V_trace[k, :] = V
                (s, a, r, next_s) = sample_one_uniform_from_mdp(self.mdp)
                self.model.update(np.array([(s, a, r, next_s)]))

                for d in range(len(V_queue)):
                        count_queue[d][a, s] += 1
                        lr = 1 / count_queue[d][a, s]
                        PV_queue[d][a, s] += lr * (V_queue[d][next_s] - PV_queue[d][a, s])

                Phat = self.model.get_P_hat()
                if k == 0:
                    Pbar = np.copy(Phat)
                if k % planning_interval == 0:
                    # if k >= num_unused_V * val_queue_interval:
                    if len(V_queue) > num_unused_V:
                        M = (np.array(V_queue)[:-num_unused_V, :]).T
                        PM = np.moveaxis(np.array(PV_queue), 0, 2)[:, :, :-num_unused_V]
                        
                        start_time = time.time()
                        Pbar, _ = MECVI_Control.calc_Pbar_Control(M, PM, Phat, beta, zero_dual_params[:, :, :M.shape[1]])
                        self.opt_time_trace.append(time.time() - start_time)
                        self.Phat_trace.append(Phat)
                        self.Pbar_trace.append(Pbar)
                        self.model_iter_trace.append(k)
                        
                        V = value_iteration(Pbar, R.reshape((num_states * num_actions)), discount, err=1e-6, max_iteration=100000)
                        policy = (R.reshape((num_states * num_actions)) + discount * Pbar.reshape((-1, num_states)) @ V).reshape((-1, num_states)).argmax(axis=0)
                    else:
                        V = value_iteration(Phat, R.reshape((num_states * num_actions)), discount, err=1e-6, max_iteration=100000)
                        policy = (R.reshape((num_states * num_actions)) + discount * Phat.reshape((-1, num_states)) @ V).reshape((-1, num_states)).argmax(axis=0)

                if k % val_queue_interval == 0:
                    # print(np.linalg.norm(V, ord=2))
                    new_U = np.copy(V)
                    if len(V_queue) > 0 and self.order > 1:
                        U = (np.array(V_queue)[-self.order+1:]).T / basis_norm
                        new_U = new_U - U @ U.T @ new_U
                    new_U = basis_norm * new_U / (np.linalg.norm(new_U, ord=2) + 1e-7)
                    V_queue.append(new_U)
                    # PV_queue.append((self.mdp.P().reshape((-1, num_states)) @ new_U).reshape((num_actions, num_states)))
                    if len(PV_queue) > 0:
                        PV_queue.append(np.copy(PV_queue[-1]))
                    else:
                        PV_queue.append(np.zeros((num_actions, num_states)))
                    count_queue.append(np.zeros((num_actions, num_states)))
                    if len(V_queue) > self.order + num_unused_V:
                        V_queue.pop(0)
                        PV_queue.pop(0)
                        count_queue.pop(0)

                
                


                self.policy_trace[k, :] = policy
                x = np.zeros((self.order))
                if len(V_queue) > num_unused_V:
                    M = (np.array(V_queue)[:-num_unused_V, :]).T
                    PM = np.moveaxis(np.array(PV_queue), 0, 2)[:, :, :-num_unused_V].reshape((num_states * num_actions, -1))
                    true_PM = self.mdp.P().reshape((-1, num_states)) @ M
                    x = np.mean(np.abs(PM - true_PM), axis=0)
                self.q_error_trace[k, :] = x
                self.V_pi_trace[k,:] = get_policy_value_mdp(self.mdp, policy, err=1e-6, max_iteration=100000)
                self.V_trace[k, :] = V

    def run(self, num_iteration, num_unused_V, beta, planning_interval, val_queue_interval,
             policy_filename, value_filename, corrected_model_filename, uncorrected_model_filename, opt_time_filename, iter_filename, save_to_file=True):
        self.train(num_iteration, num_unused_V, beta, planning_interval, val_queue_interval)
        if save_to_file:
            np.save(value_filename, self.V_pi_trace)
            np.save(policy_filename, self.policy_trace)
            np.save(uncorrected_model_filename, np.array(self.Phat_trace))
            np.save(corrected_model_filename, np.array(self.Pbar_trace))
            np.save(opt_time_filename, np.array(self.opt_time_trace))
            np.save(iter_filename, np.array(self.model_iter_trace))
        return self.V_trace, self.V_pi_trace, self.policy_trace, self.q_error_trace
