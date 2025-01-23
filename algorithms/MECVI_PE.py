import sys
import time



# setting path
sys.path.append('../splitting_tabular')

from env.Maze import Maze33
from algorithms.VI_PE import VI_PE
from model.IdentitySmoothedModel import IdentitySmoothedModel
from matplotlib import pyplot as plt
from algorithms.MSVI_PE import MSVI_PE
from env.CliffWalk import CliffWalk
from model.LocallySmoothedModel import LocallySmoothedModel
 
from scipy.optimize import minimize
from scipy.special import logsumexp

import numpy as np
from tqdm import tqdm
from rl_utilities import get_optimal_policy_mdp, get_policy_value, get_policy_value_mdp


class MECVI_PE:
    def __init__(self, mdp, policy, Phat, order, proj_measure="KL", beta=np.inf):
        self.order = order
        self.mdp = mdp
        self.policy = policy
        self.Phat = Phat
        self.proj_measure = proj_measure
        self.beta = beta

    def train(self, num_iteration):
        num_states = self.mdp.num_states()
        num_actions = self.mdp.num_actions()
        R = self.mdp.R()
        P = self.mdp.P()
        Phat = self.Phat

        r_pi = R[self.policy, np.arange(num_states)]
        P_pi = P[self.policy, np.arange(num_states), :]
    

        self.V_trace = np.zeros((num_iteration, num_states))
        self.PpiV_trace = np.zeros((num_iteration, num_states))
        V = np.zeros((num_states))
        dual_params = np.zeros((num_states, self.order))

        initial_M = np.zeros((num_states, self.order))
        initial_PM = P_pi @ initial_M    

        with tqdm(iter(range(num_iteration)), desc=f"MECVI{self.order}", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                # print(k)

                M, PM = initial_M.copy(), initial_PM.copy()
                if k > 0:
                    M[:, :min(self.order, k)] = self.V_trace[max(k - self.order, 0):k, :].T
                    PM[:, :min(self.order, k)] = self.PpiV_trace[max(k - self.order, 0):k, :].T


                start_time = time.time()
                Pbar, dual_params = MECVI_PE.calc_Pbar_PE(M, PM, Phat, self.policy, self.beta, dual_params)
                # if k >= self.order - 1:
                #     Pbar, dual_params = MECVI_PE.calc_Pbar_PE(M, PM, Phat, self.policy, self.beta, dual_params)
                # else:
                #     Pbar, _ = MECVI_PE.calc_Pbar_PE(M, PM, Phat, self.policy, self.beta, dual_params[:, :k+1])
                # print("Optimization Time:", time.time() - start_time)
                
                start_time = time.time()
                V = get_policy_value(Pbar, R, self.mdp.discount(), self.policy, err=1e-10)

                self.V_trace[k, :] = V
                self.PpiV_trace[k, :] = P_pi @ V
                # print("Planning Time:", time.time() - start_time)

    @staticmethod
    def calc_Pbar_PE(M, PM, Phat, policy, beta, initial_dual_params):
        num_states = Phat.shape[2]
        Phat_pi = Phat[policy, np.arange(num_states), :]
        Pbar_pi, dual_params =  MECVI_PE.calc_Pbar_mat(M, PM, Phat_pi, beta, initial_dual_params)
        Pbar = np.copy(Phat)
        Pbar[policy, np.arange(num_states), :] = Pbar_pi

        return Pbar, dual_params


    # @staticmethod
    # def dual_loss_hessp(v_flatten, u_flatten, M, Phat_pi, Y):
    #     n = M.shape[0]
    #     q = M.shape[1]

    #     V = v_flatten.reshape((n, q))
    #     U = u_flatten.reshape((n, q))

    #     Lambda = logsumexp(V @ M.T, b=Phat_pi, axis=1, keepdims=True)
    #     Q = np.multiply(Phat_pi, np.exp((V @ M.T - Lambda) * (Phat_pi > 0) ))

    #     out_matrix = (np.sum((U @ M.T) * Q, axis = 1, keepdims= True) * Q ) @ M
    #     out_matrix -= ((U @ M.T) * Q) @ M
    #     out_flatten = out_matrix.reshape(-1)
    #     out_flatten = - out_flatten

    #     return out_flatten

    @staticmethod
    def dual_loss_jac(v_flatten, M, Phat_mat, PM_mat, beta):
        n = Phat_mat.shape[0]
        q = M.shape[1]
        V = v_flatten.reshape((n, q))
        Lambda = logsumexp(V @ M.T, b=Phat_mat, axis=1, keepdims=True)
        Q_lambdda = np.multiply(Phat_mat, np.exp((V @ M.T - Lambda) * (Phat_mat > 0) ))
        grad_matrix = - PM_mat + Q_lambdda @ M + (2 / beta) * V
        grad_flatten = grad_matrix.reshape(-1) / n
        return grad_flatten
    
    @staticmethod
    def dual_loss(v_flatten, M, Phat_mat, PM_mat, beta):
        n = Phat_mat.shape[0]
        q = M.shape[1]
        V = v_flatten.reshape((n, q))
        negloss_vector = np.sum(np.multiply(V, PM_mat), axis=1)
        negloss_vector -= logsumexp(V @ M.T, b=Phat_mat, axis=1)
        negloss_vector += (1 / beta) * np.linalg.norm(V, ord=2, axis=1) ** 2
        loss = - (np.mean(negloss_vector) + np.exp(-1))
        return loss

    @staticmethod
    def calc_Pbar_from_dual(Phat, policy, dual_params, M):
        num_states = Phat.shape[2]
        V = dual_params
        Phat_pi = Phat[policy, np.arange(num_states), :]
        Lambda = logsumexp(V @ M.T, b=Phat_pi, axis=1, keepdims=True)
        Pbar_pi = np.multiply(Phat_pi, np.exp((V @ M.T - Lambda) * (Phat_pi > 0)))
        Pbar = np.copy(Phat)
        Pbar[policy, np.arange(num_states), :] = Pbar_pi
        return Pbar

    @staticmethod
    def calc_Pbar_mat(M, PM_mat, Phat_mat, beta, initial_dual_params_mat):
        n = Phat_mat.shape[0]
        q = M.shape[1]

        col_means = np.mean(M, axis=0, keepdims=True)
        col_stds = np.maximum(np.std(M, axis=0, keepdims=True), 1e-4 * np.ones((q)))
        M = (M - col_means) / col_stds
        PM_mat = (PM_mat - col_means) / col_stds
        
        v0_flatten = initial_dual_params_mat.reshape(-1)
        res = minimize(MECVI_PE.dual_loss, x0=v0_flatten, jac=MECVI_PE.dual_loss_jac, args=(M, Phat_mat, PM_mat, beta))
        v_flatten = res.x
        V = v_flatten.reshape((n, q))
        Lambda = logsumexp(V @ M.T, b=Phat_mat, axis=1, keepdims=True)
        Pbar_mat = np.multiply(Phat_mat, np.exp((V @ M.T - Lambda) * (Phat_mat > 0)))
        return Pbar_mat, V


    def run(self, num_iteration, output_filename, save_to_file=True):
        self.train(num_iteration)
        if save_to_file:
            np.save(output_filename, self.V_trace)
        return self.V_trace



if __name__ == "__main__":
    alpha_vals = [0.1, 0.5, 1]
    order_vals = [1, 2, 3]
    # alpha_vals = [0.1, 0.5]
    # order_vals = [2]
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
    for j in range(len(alpha_vals)):
        alpha = alpha_vals[j]
        Phat = LocallySmoothedModel.get_P_hat_using_P(mdp.P(), alpha)

        vi_pe = VI_PE(mdp, policy)
        v_trace= vi_pe.run(num_iterations, "", save_to_file=False)
        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        vi_value_errors[j, :] = value_errors / np.linalg.norm(true_V, ord=1)

        msvi = MSVI_PE(mdp, policy, Phat)
        v_trace = msvi.run(num_iterations, "", save_to_file=False)
        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        msvi_value_errors[j, :] = value_errors / np.linalg.norm(true_V, ord=1)

        
        for i in range(len(order_vals)):
            mecvi_order = order_vals[i]
            mecvi = MECVI_PE(mdp, policy, Phat, mecvi_order, proj_measure="KL", beta=beta)
            v_trace = mecvi.run(num_iterations, "", save_to_file=False)
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

    plt.suptitle("IdentitySmoothing in Cliffwalk (gamma = 0.99, success=0.9) - KL Projection", size = 30)
    plt.savefig("output/tmp.png")

    # plt.figure()

    # plt.plot(np.arange(num_iterations), vi_value_errors, label=rf"VI")
    # plt.plot(np.arange(num_iterations), msvi_value_errors, label=rf"OS-VI ($\lambda$ = {alpha})")
    # for i in range(len(order_vals)):    
    #     mecvi_order = order_vals[i]
    #     query_nums = np.arange(0, num_iterations * mecvi_order, step=mecvi_order)
    #     plt.plot(query_nums, mecvi_value_errors[i, :len(query_nums)], label=rf"MEC-VI ($\lambda$ = {alpha}, ord={mecvi_order})")
    
    
    # plt.legend()
    # plt.yscale("log")
    # plt.ylim(1e-8, 1)
    # plt.xlim(0, 100)
    # plt.xlabel("queries")
    # plt.savefig("tmp2.png")

