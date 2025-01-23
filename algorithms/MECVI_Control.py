import sys
import time



# setting path
sys.path.append('../splitting_tabular')

from algorithms.MECVI_PE import MECVI_PE
from algorithms.MSVI_Control import MSVI_Control
from algorithms.VI_Control import VI_Control
from env.Maze import Maze33
from algorithms.VI_PE import VI_PE
from model.IdentitySmoothedModel import IdentitySmoothedModel
from matplotlib import pyplot as plt
from algorithms.MSVI_PE import MSVI_PE
from env.CliffWalk import CliffWalk
from model.LocallySmoothedModel import LocallySmoothedModel
 
from scipy.optimize import minimize
from scipy.special import logsumexp
from tqdm import tqdm

import numpy as np
from rl_utilities import get_optimal_policy, get_optimal_policy_mdp, get_policy_value, get_policy_value_mdp


class MECVI_Control:
    def __init__(self, mdp, Phat, order, proj_measure="KL", beta=np.inf):
        self.order = order
        self.mdp = mdp
        self.Phat = Phat
        self.proj_measure = proj_measure
        self.beta = beta

    def train(self, num_iteration):
        num_states = self.mdp.num_states()
        num_actions = self.mdp.num_actions()
        R = self.mdp.R()
        P = self.mdp.P()
        Phat = self.Phat

        r_mat = R.reshape((num_actions * num_states))
        P_mat = P.reshape((num_actions * num_states, num_states))
        Phat_mat = Phat.reshape((num_actions * num_states, num_states))
    
        self.V_trace = np.zeros((num_iteration, num_states))
        self.PV_mat_trace = np.zeros((num_iteration, num_actions * num_states))
        self.policy_trace = np.zeros((num_iteration, num_states))
        V = np.zeros((num_states))
        optimal_policy = np.zeros((num_states))
        dual_params = np.zeros((num_states * num_actions, self.order))

        initial_M = np.zeros((num_states, self.order))
        initial_PM = P_mat @ initial_M    

        with tqdm(iter(range(num_iteration)), desc=f"MECVI{self.order}", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                # print(k)
                
                self.policy_trace[k, :] = optimal_policy

                M, PM_mat = initial_M.copy(), initial_PM.copy()
                M[:, :min(self.order, k)] = self.V_trace[max(k - self.order, 0):k, :].T
                PM_mat[:, :min(self.order, k)] = self.PV_mat_trace[max(k - self.order, 0):k, :].T

                
                start_time = time.time()
                Pbar_mat, dual_params = MECVI_PE.calc_Pbar_mat(M, PM_mat, Phat_mat, self.beta, dual_params)
                # if k >= self.order - 1:
                #     Pbar_mat, dual_params = MECVI_PE.calc_Pbar_mat(M, PM_mat, Phat_mat, self.beta, dual_params)
                # else:
                #     Pbar_mat, _ = MECVI_PE.calc_Pbar_mat(M, PM_mat, Phat_mat, self.beta, dual_params[:, :k+1])

                
                # print("Optimization Time:", time.time() - start_time)
                
                Pbar = Pbar_mat.reshape((num_actions, num_states, num_states))

                start_time = time.time()
                optimal_policy = get_optimal_policy(Pbar, R, self.mdp.discount(), num_states, num_actions, err=1e-6, max_iterations=100000)
                V = get_policy_value(Pbar, R, self.mdp.discount(), optimal_policy, err=1e-10)

                self.V_trace[k, :] = V

                # Query
                self.PV_mat_trace[k, :] = P_mat @ V

                # print("Planning Time:", time.time() - start_time)

    @staticmethod
    def calc_Pbar_Control(M, PM, Phat, beta, initial_dual_params):
        num_states, num_actions = Phat.shape[2], Phat.shape[0]
        order = M.shape[1]
        PM_mat = PM.reshape((num_states * num_actions, order))
        Phat_mat = Phat.reshape((num_actions * num_states, num_states))
        initial_dual_params_mat = initial_dual_params.reshape((num_states * num_actions, order))
        Pbar_mat, dual_params_mat =  MECVI_PE.calc_Pbar_mat(M, PM_mat, Phat_mat, beta, initial_dual_params_mat)
        Pbar = Pbar_mat.reshape((num_actions, num_states, num_states))
        dual_params = dual_params_mat.reshape((num_actions, num_states, order))
        return Pbar, dual_params


    def run(self, num_iteration, policy_filename, value_filename, save_to_file=True):
        self.train(num_iteration)
        if save_to_file:
            np.save(policy_filename, self.policy_trace)
            np.save(value_filename, self.V_trace)
        return self.V_trace, self.policy_trace



if __name__ == "__main__":
    # alpha_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    # order_vals = [1, 2, 3, 4, 5, 6]
    alpha_vals = [0.1, 0.5, 1]
    order_vals = [1, 2, 3]
    beta = np.inf
    
    num_iterations = 20
    mdp = CliffWalk(0.9, 0.9)
    policy = get_optimal_policy_mdp(mdp)
    
    true_V = get_policy_value_mdp(mdp, policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V

    vi_value_errors = np.ones((len(alpha_vals), num_iterations))
    msvi_value_errors = np.ones((len(alpha_vals), num_iterations))
    mecvi_value_errors = np.ones((len(alpha_vals), len(order_vals), num_iterations))

    msvi_v_pi = np.ones((len(alpha_vals), num_iterations))
    mecvi_v_pi = np.ones((len(alpha_vals), len(order_vals), num_iterations))

    
    for j in range(len(alpha_vals)):
        alpha = alpha_vals[j]
        Phat = LocallySmoothedModel.get_P_hat_using_P(mdp.P(), alpha)

        vi_control = VI_Control(mdp)
        v_trace, _ = vi_control.run(num_iterations, "", "", save_to_file=False)
        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        vi_value_errors[j, :] = value_errors / np.linalg.norm(true_V, ord=1)

        msvi = MSVI_Control(mdp, Phat)
        v_trace, policy_trace = msvi.run(num_iterations, "", "", save_to_file=False)
        for k in range(num_iterations):
                msvi_v_pi[j, k] = get_policy_value(mdp.P(), mdp.R(), mdp.discount(), policy_trace[k, :].astype(int))[0]                
        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        msvi_value_errors[j, :] = value_errors / np.linalg.norm(true_V, ord=1)

        
        for i in range(len(order_vals)):
            mecvi_order = order_vals[i]
            mecvi = MECVI_Control(mdp, Phat, mecvi_order, proj_measure="KL", beta=beta)
            v_trace, policy_trace = mecvi.run(num_iterations, "", "", save_to_file=False)
            for k in range(num_iterations):
                mecvi_v_pi[j, i, k] = get_policy_value(mdp.P(), mdp.R(), mdp.discount(), policy_trace[k, :].astype(int))[0]
            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
            mecvi_value_errors[j, i, :] = value_errors / np.linalg.norm(true_V, ord=1)
            print("alpha", alpha, "- order", mecvi_order)


    fig, axs = plt.subplots(2, len(alpha_vals))
    fig.set_size_inches(9 * len(alpha_vals), 12)
    for j in range(len(alpha_vals)):
        alpha = alpha_vals[j]
        axs[0, j].plot(np.arange(num_iterations), vi_value_errors[j, :], label=rf"VI")
        axs[0, j].plot(np.arange(num_iterations), msvi_value_errors[j, :], label=rf"OS-VI ($\lambda$ = {alpha})")
        for i in range(len(order_vals)):    
            mecvi_order = order_vals[i]
            axs[0, j].plot(np.arange(num_iterations), mecvi_value_errors[j, i, :], label=rf"MEC-VI ($\lambda$ = {alpha}, ord={mecvi_order})")
    
        axs[0, j].legend()
        axs[0, j].set_yscale("log")
        axs[0, j].set_ylim(1e-5, 10)
        axs[0, j].set_xlabel("iterations")
    
    for j in range(len(alpha_vals)):
        alpha = alpha_vals[j]
        axs[1, j].plot(np.arange(num_iterations), vi_value_errors[j, :], label=rf"VI")
        axs[1, j].plot(np.arange(num_iterations), msvi_value_errors[j, :], label=rf"OS-VI ($\lambda$ = {alpha})")
        for i in range(len(order_vals)):    
            mecvi_order = order_vals[i]
            query_nums = np.arange(0, num_iterations * mecvi_order, step=mecvi_order)
            axs[1, j].plot(query_nums, mecvi_value_errors[j, i, :len(query_nums)], label=rf"MEC-VI ($\lambda$ = {alpha}, ord={mecvi_order})")

        axs[1, j].legend()
        axs[1, j].set_yscale("log")
        axs[1, j].set_ylim(1e-5, 10)
        axs[1, j].set_xlim(0, 100)
        axs[1, j].set_xlabel("queries")

    plt.suptitle("UniformSmoothing in Cliffwalk (gamma = 0.99, success=0.9) - KL-L1 Projection", size = 30)
    plt.savefig("output/tmp.png")

    #### V_pi plot
    fig, axs = plt.subplots(2, len(alpha_vals))
    fig.set_size_inches(9 * len(alpha_vals), 12)
    for j in range(len(alpha_vals)):
        alpha = alpha_vals[j]
        axs[0, j].plot(np.arange(num_iterations), msvi_v_pi[j, :], label=rf"OS-VI ($\lambda$ = {alpha})")
        for i in range(len(order_vals)):    
            mecvi_order = order_vals[i]
            axs[0, j].plot(np.arange(num_iterations), mecvi_v_pi[j, i, :], label=rf"MEC-VI ($\lambda$ = {alpha}, ord={mecvi_order})")
    
        axs[0, j].legend()
        # axs[0, j].set_yscale("log")
        # axs[0, j].set_ylim(1e-5, 10)
        axs[0, j].set_xlabel("iterations")
    
    
    plt.suptitle("UniformSmoothing in Cliffwalk (gamma = 0.99, success=0.9) - KL-L1 Projection", size = 30)
    plt.savefig("output/tmp_vi_pi.png")

