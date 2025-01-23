from matplotlib import pyplot as plt, ticker
import scipy
import seaborn as sns
import numpy as np

from env.CliffWalk import CliffWalk
from rl_utilities import get_optimal_policy_mdp, get_policy_value_mdp

# np.seterr(all='raise')

alpha_vals = [0.1, 0.5, 1]
num_trials = 20
num_iterations = 150000
plot_every = 2000

initial_dist = np.zeros((36))
initial_dist[0] = 1

exp_dir = f"./output/8a0f_cliffwalk/exp_control_sample"

# Plotting
params = {'font.size': 26,
            'axes.labelsize': 26, 'axes.titlesize': 26, 'legend.fontsize': 22, "axes.titlepad":12,
            'xtick.labelsize': 22, 'ytick.labelsize': 22, 'lines.linewidth': 3, 'axes.linewidth': 2}
plt.rcParams.update(params)



mdp = CliffWalk(0.9, 0.9)

def kl_model_error(phats, p):
    populated_p = np.expand_dims(p, axis=0)
    eps_model = scipy.special.kl_div(populated_p, phats)
    eps_model = np.sum(eps_model, axis=3)
    return np.sum(np.sqrt(eps_model), axis=(1, 2))

def tv_model_error(phats, p):
    populated_p = np.expand_dims(p, axis=0)
    return np.mean(np.sum(np.abs(populated_p - phats), axis=(3)), axis=(1, 2))
     


def plot_alg(ax, alg_dir, label, color_code, linestyle):
    value_per_trial = []
    for trial in range(num_trials):  
        with open(f'{alg_dir}/value_trace_{trial}.npy', 'rb') as f:
            value_trace = np.load(f)
            value_per_trial.append(value_trace.dot(initial_dist))

    mean = np.array(value_per_trial).mean(axis=0)
    stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

    ax.plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=label, linestyle=linestyle, color=cmap[color_code])
    ax.fill_between(x=np.arange(num_iterations)[::plot_every],
                        y1=(mean - stderr)[::plot_every],
                        y2=(mean + stderr)[::plot_every],
                        alpha=0.1,
                        color=cmap[color_code]
                        )
    
def plot_phat_error(ax, alg_dir, label, color_code):

    with open(f'{alg_dir}/iter_trace_0.npy', 'rb') as f:
            iter_trace = np.load(f)

    phat_error_per_trial = []
    for trial in range(num_trials):  
        with open(f'{alg_dir}/phat_trace_{trial}.npy', 'rb') as f:
            phat_trace = np.load(f)
            phat_error_per_trial.append(tv_model_error(phat_trace, mdp.P()))

    mean = np.array(phat_error_per_trial).mean(axis=0)
    stderr = np.array(phat_error_per_trial).std(axis=0) / np.sqrt(num_trials)

    ax.plot(iter_trace, mean, label=label, linestyle="dashed", color=cmap[color_code])
    ax.fill_between(x=iter_trace,
                        y1=(mean - stderr),
                        y2=(mean + stderr),
                        alpha=0.1,
                        color=cmap[color_code]
                        )    
def plot_opt_time(ax, alg_dir, label, color_code):

    with open(f'{alg_dir}/iter_trace_0.npy', 'rb') as f:
            iter_trace = np.load(f)
    
    opt_time_per_trial = []
    for trial in range(num_trials):  
        with open(f'{alg_dir}/pbar_trace_{trial}.npy', 'rb') as f:
            pbar_trace = np.load(f)
            opt_time_per_trial.append(tv_model_error(pbar_trace, mdp.P()))

    mean = np.array(opt_time_per_trial).mean(axis=0)
    stderr = np.array(opt_time_per_trial).std(axis=0) / np.sqrt(num_trials)

    ax.plot(iter_trace, mean, label=label, linestyle="solid", color=cmap[color_code])
    ax.fill_between(x=iter_trace,
                        y1=(mean - stderr),
                        y2=(mean + stderr),
                        alpha=0.1,
                        color=cmap[color_code]
                        )   

fig, axs = plt.subplots(1, 3, sharey=True)
plt.subplots_adjust(left=0.05,
                        bottom=0.26,
                        right=0.99,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)
fig.set_figwidth(20)
fig.set_figheight(7)
fig.set_dpi(80)
cmap = sns.color_palette(palette="colorblind", n_colors=8)
subplot_titles = ["Low Model Error", "Medium Model Error", "High Model Error"]

for i in range(len(alpha_vals)):
    alpha = alpha_vals[i]
    plot_alg(axs[i], f"{exp_dir}/mecdyna1_control_{alpha}", rf"MoCoDyna ($d$=1)", 7, "solid")
    plot_alg(axs[i], f"{exp_dir}/mecdyna2_control_{alpha}", rf"MoCoDyna ($d$=2)", 0, "solid")
    plot_alg(axs[i], f"{exp_dir}/mecdyna3_control_{alpha}", rf"MoCoDyna ($d$=3)", 5, "solid")
    plot_alg(axs[i], f"{exp_dir}/msdyna_control_{alpha}", "OS-Dyna", 2, "dashdot")
    plot_alg(axs[i], f"{exp_dir}/qlearning_control", "QLearning", 2, "dotted")
    plot_alg(axs[i], f"{exp_dir}/dyna_control_{alpha}", "Dyna", 1, "dashed")

    policy = get_optimal_policy_mdp(mdp)
    true_V = get_policy_value_mdp(mdp, policy).dot(initial_dist)
    axs[i].axhline(y=true_V, linestyle="dashed", color="black")

    if i == 0:
        axs[0].set_ylabel(r'$V^{\pi_t}(0)$')
        # axs[0].set_ylabel("Return")
    if i == 0:
        axs[0].set_xlabel("Environment Samples")
    axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
    axs[i].title.set_text(subplot_titles[i])
    axs[i].grid()
handles, labels = axs[0].get_legend_handles_labels()
order = [i for i in range(6)]
fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower center", ncol=6)

plt.savefig(f"{exp_dir}/control_sample.png", bbox_inches="tight")
plt.savefig(f"{exp_dir}/control_sample.pdf", bbox_inches="tight")
