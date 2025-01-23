from matplotlib import pyplot as plt, ticker
import seaborn as sns
import numpy as np

from env.CliffWalk import CliffWalk
from rl_utilities import get_optimal_policy_mdp, get_policy_value_mdp

alpha_vals = [0.1, 0.5, 1]
num_trials = 20
num_iterations = 10000
plot_every = 100

initial_dist = np.zeros((36))
initial_dist[0] = 1

exp_dir = f"./output/8a0f_cliffwalk/exp_pe_sample"

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
                        wspace=0.1,
                        hspace=0.1)
fig.set_figwidth(20)
fig.set_figheight(7)
fig.set_dpi(150)
cmap = sns.color_palette(palette="colorblind", n_colors=8)
subplot_titles = ["Low Model Error", "Medium Model Error", "High Model Error"]

mdp = CliffWalk(0.9, 0.9)
policy = get_optimal_policy_mdp(mdp)
true_V = get_policy_value_mdp(mdp, policy)
print(true_V.shape)
true_trace = np.zeros((num_iterations, mdp.num_states()))
true_trace[np.arange(num_iterations), :] = true_V

def plot_alg(ax, alg_dir, label, color_code, linestyle):
    value_error_per_trial = []
    for trial in range(num_trials):  
        with open(f'{alg_dir}/V_trace_{trial}.npy', 'rb') as f:
            v_trace = np.load(f)
            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
            value_error_per_trial.append(value_errors / np.sum(np.abs(true_V)))

    mean = np.array(value_error_per_trial).mean(axis=0)
    stderr = np.array(value_error_per_trial).std(axis=0) / np.sqrt(num_trials)

    ax.plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=label, linestyle=linestyle, color=cmap[color_code])
    ax.fill_between(x=np.arange(num_iterations)[::plot_every],
                        y1=(mean - stderr)[::plot_every],
                        y2=(mean + stderr)[::plot_every],
                        alpha=0.1,
                        color=cmap[color_code]
                        )

for i in range(len(alpha_vals)):
    alpha = alpha_vals[i]
    plot_alg(axs[i], f"{exp_dir}/mecdyna1_pe_{alpha}", rf"MoCoDyna ($d$=1)", 7, "solid")
    plot_alg(axs[i], f"{exp_dir}/mecdyna2_pe_{alpha}", rf"MoCoDyna ($d$=2)", 4, "solid")
    plot_alg(axs[i], f"{exp_dir}/mecdyna3_pe_{alpha}", rf"MoCoDyna ($d$=3)", 5, "solid")
    plot_alg(axs[i], f"{exp_dir}/msdyna_pe_{alpha}", "OS-Dyna", 2, "dashdot")
    plot_alg(axs[i], f"{exp_dir}/tdlearning_pe", "TD-Learning", 0, "dotted")
    plot_alg(axs[i], f"{exp_dir}/dyna_pe_{alpha}", "Dyna", 1, "dashed")


    if i == 0:
        axs[0].set_ylabel(r'$||V^\pi - V_k||$')
    if i == 1:
        axs[1].set_xlabel("Environment Samples (t)")
    axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
    axs[i].title.set_text(subplot_titles[i])
    axs[i].set_yscale("log")
    axs[i].grid()
handles, labels = axs[0].get_legend_handles_labels()
order = [i for i in range(6)]
fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower center", ncol=6)

plt.savefig(f"{exp_dir}/pe_sample.png", bbox_inches="tight")
plt.savefig(f"{exp_dir}/pe_sample.pdf", bbox_inches="tight")
