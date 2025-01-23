from matplotlib import pyplot as plt, ticker
import seaborn as sns
import numpy as np

from env.CliffWalk import CliffWalk
from model.LocallySmoothedModel import LocallySmoothedModel
from rl_utilities import get_optimal_policy_mdp, get_policy_value, get_policy_value_mdp

alpha_vals = [0.1, 0.5, 1]
num_iterations = 15
num_states = 36

# initial_dist = np.zeros((36))
# initial_dist[0] = 1
initial_dist = np.ones((36)) / 36

mdp = CliffWalk(0.9, 0.9)
policy = get_optimal_policy_mdp(mdp)
true_Vfunc = get_policy_value_mdp(mdp, policy)
true_V = get_policy_value_mdp(mdp, policy).dot(initial_dist)

exp_dir = f"./output/8a0f_cliffwalk/exp_pe_vector0"
model_class = LocallySmoothedModel


# Plotting
params = {'font.size': 26,
            'axes.labelsize': 26, 'axes.titlesize': 26, 'legend.fontsize': 22, "axes.titlepad":12,
            'xtick.labelsize': 22, 'ytick.labelsize': 22, 'lines.linewidth': 3, 'axes.linewidth': 2}
plt.rcParams.update(params)

fig, axs = plt.subplots(1, 3, sharey=True)
plt.subplots_adjust(left=0.05,
                        bottom=0.26,
                        right=0.99,
                        top=0.9,
                        wspace=0.05,
                        hspace=0.1)
fig.set_figwidth(20)
fig.set_figheight(7)
fig.set_dpi(100)
cmap = sns.color_palette(palette="colorblind", n_colors=8)
subplot_titles = ["Low Model Error", "Medium Model Error", "High Model Error"]

def plot_alg_vpi(ax, alg_dir, label, color_code, linestyle):
    value_per_trial = []  
    with open(f'{alg_dir}/policy_trace.npy', 'rb') as f:
        policy_trace = np.load(f)
        value_trace = np.zeros((num_iterations, num_states))
        for i in range(num_iterations):
            value_trace[i, :] = get_policy_value_mdp(mdp, policy_trace[i, :].astype(int))   
        yvalue = value_trace.dot(initial_dist)

    ax.plot(np.arange(num_iterations), yvalue, label=label, linestyle=linestyle, color=cmap[color_code])

def plot_alg_v(ax, alg_dir, label, color_code, linestyle):
    value_per_trial = []  
    with open(f'{alg_dir}/V_trace.npy', 'rb') as f:
        value_trace = np.load(f)
    yvalue = np.linalg.norm(value_trace - true_Vfunc, ord=1, axis=1)/np.linalg.norm(true_Vfunc, ord=1)

    ax.plot(np.arange(num_iterations), yvalue, label=label, linestyle=linestyle, color=cmap[color_code])
    
for i in range(len(alpha_vals)):
    alpha = alpha_vals[i]

    plot_alg_v(axs[i], f"{exp_dir}/mecvi1_pe_{alpha}", rf"MoCoVI ($d$=1)", 7, "solid")
    plot_alg_v(axs[i], f"{exp_dir}/mecvi2_pe_{alpha}", rf"MoCoVI ($d$=2)", 4, "solid")
    plot_alg_v(axs[i], f"{exp_dir}/mecvi3_pe_{alpha}", rf"MoCoVI ($d$=3)", 5, "solid")
    plot_alg_v(axs[i], f"{exp_dir}/msvi_pe_{alpha}", "OS-VI", 2, "dashdot")
    plot_alg_v(axs[i], f"{exp_dir}/vi_pe", "VI", 0, "dotted")
    
    Phat = model_class.get_P_hat_using_P(mdp.P(), alpha_vals[i])
    value = get_policy_value(Phat, mdp.R(), mdp.discount(), policy)
    value_error = np.linalg.norm(value - true_Vfunc, ord=1)
    value_error = value_error / np.linalg.norm(true_Vfunc, ord=1)
    axs[i].axhline(y=value_error, label=r"Pure MBRL".format(alpha_vals[i]),
                    linestyle="dashed", color=cmap[1])


    if i == 0:
        axs[0].set_ylabel(r'Normalized $||V_k - V^{\pi}||$')
    if i == 1:
        axs[1].set_xlabel("Iteration (k)")
    axs[i].set_yscale("log")
    axs[i].set_ylim(1e-3, 2)
    axs[i].set_xlim(0, 15)
    axs[i].set_xticks([i for i in range(0, 16, 2)])
    axs[i].title.set_text(subplot_titles[i])
    axs[i].grid()
handles, labels = axs[0].get_legend_handles_labels()
order = [i for i in range(6)]
fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower center", ncol=6)

plt.savefig(f"{exp_dir}/pe_vector.png", bbox_inches="tight")
plt.savefig(f"{exp_dir}/pe_vector.pdf", bbox_inches="tight")
