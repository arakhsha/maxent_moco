import argparse
import numpy as np
import yaml
import shutil
import random
from algorithms.MECDyna_PE import MECDyna_PE
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from multiprocessing import Pool
from algorithms.MSDyna_PE import MSDyna_PE
from algorithms.Dyna_PE import Dyna_PE
from algorithms.TDLearning_PE import TDLearning_PE
from rl_utilities import get_optimal_policy_mdp
from utilities import setup_problem, setup_alg_output_dir, get_exp_dir, get_default_alg_output_dir
from LearningRate import LearningRate

ROOT_OUTPUT_DIR = "./output"

def run_td_learning(inputs):
    mdp, policy, config, config_path, num_iterations, trial, exp_dir, model_class = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    tdlearning_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "tdlearning_pe", smoothing_param=None,
                                              exp_dir=exp_dir)
    tdlearning_pe = TDLearning_PE(mdp, policy)
    lr_scheduler = LearningRate(config["exp_pe_lr_type"],
                                config["exp_pe_td_learning_pe_lr"],
                                config["exp_pe_td_learning_pe_delay"],
                                config["exp_pe_td_gamma"])
    shutil.copyfile(src=config_path, dst=f"{tdlearning_out_dir}/config.yaml")
    tdlearning_pe.run(num_iterations, lr_scheduler=lr_scheduler,
                      output_filename=f"{tdlearning_out_dir}/V_trace_{trial}.npy")

def run_dyna_pe(inputs):
    mdp, policy, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        dyna_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "dyna_pe", smoothing_param=alpha, exp_dir=exp_dir)
        dyna_pe = Dyna_PE(mdp, policy, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        shutil.copyfile(src=config_path, dst=f"{dyna_out_dir}/config.yaml")
        dyna_pe.run(num_iterations, output_filename=f"{dyna_out_dir}/V_trace_{trial}.npy")

def run_msdyna_pe(inputs):
    mdp, policy, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        msdyna_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "msdyna_pe", smoothing_param=alpha, exp_dir=exp_dir)
        msdyna_pe = MSDyna_PE(mdp, policy, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        lr_scheduler = LearningRate(config["exp_pe_lr_type"],
                                    config["exp_pe_msdyna_lr"][i],
                                    config["exp_pe_msdyna_lr_delay"],
                                    config["exp_pe_msdyna_gamma"][i])
        shutil.copyfile(src=config_path, dst=f"{msdyna_out_dir}/config.yaml")
        msdyna_pe.run(num_iterations, lr_scheduler=lr_scheduler,
                      output_filename=f"{msdyna_out_dir}/V_trace_{trial}.npy")

def run_mecdyna_pe(inputs):
    order, mdp, policy, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  \
        = inputs["order"], inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        beta = config[f"exp_pe_mecdyna{order}_beta"][alpha_index]
        planning_interval = config[f"exp_pe_mecdyna{order}_planning_interval"][alpha_index]
        val_queue_interval = config[f"exp_pe_mecdyna{order}_val_queue_interval"][alpha_index]
        num_unused_V = config[f"exp_pe_mecdyna{order}_num_unused_V"][alpha_index]

        mecdyna_pe = MECDyna_PE(order, mdp, policy, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        mecdyna_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", f"mecdyna{order}_pe", smoothing_param=alpha,
                                            exp_dir=exp_dir)
        print(mecdyna_out_dir)
        shutil.copyfile(src=config_path, dst=f"{mecdyna_out_dir}/config.yaml")
        mecdyna_pe.run(num_iterations,
                           num_unused_V,
                           beta,
                           planning_interval,
                           val_queue_interval,
                           output_filename=f"{mecdyna_out_dir}/V_trace_{trial}.npy")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config', help='Path  of config file')
    parser.add_argument('alg_name', help='Algorithm to run. "ALL" to run all, "None" to just plot')
    parser.add_argument('--num_trials', default=1, help='Number of trials to run')
    parser.add_argument('--alpha_index', default=-1, type=int, help='Index of alpha value to run. -1 for all.')
    parser.add_argument('--first_trial', default=0, type=int, help='Index of alpha value to run. -1 for all.')
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_pe_sample"
    exp_dir = get_exp_dir(config, "exp_pe_sample", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    policy = get_optimal_policy_mdp(mdp)
    num_trials = int(args.num_trials)
    num_iterations = config["exp_pe_sample_num_iterations"]
    alpha_vals = config["exp_pe_sample_alphas"]
    if args.alpha_index == -1:
        alpha_indices = [i for i in range(len(alpha_vals))]
    else:
        alpha_indices = [args.alpha_index]

    if config["exp_pe_sample_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_pe_sample_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        assert False

    # Running Algorithms
    if args.alg_name in ["ALL", "tdlearning_pe"]:
        tdlearning_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "tdlearning_pe", smoothing_param=None,
                                                  exp_dir=args.exp_dir)
        with Pool(24) as p:
            inputs = []
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                                "policy": policy,
                                "config": config,
                                "config_path": args.config,
                                "num_iterations": num_iterations,
                                "trial": trial,
                                "exp_dir": args.exp_dir,
                                "model_class": model_class,
                                })
            p.map(run_td_learning, inputs)

    if args.alg_name in ["ALL", "dyna_pe"]:
        for alpha in alpha_vals:
            dyna_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "dyna_pe", smoothing_param=alpha,
                                                exp_dir=args.exp_dir)
        with Pool(24) as p:
            inputs = []
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                               "policy": policy,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "alpha_indices": alpha_indices,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_dyna_pe, inputs)

    if args.alg_name in ["ALL", "msdyna_pe"]:
        for alpha in alpha_vals:
            msdyna_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "msdyna_pe", smoothing_param=alpha,
                                                  exp_dir=args.exp_dir)
        with Pool(24) as p:
            inputs = []
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                               "policy": policy,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "alpha_indices": alpha_indices,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_msdyna_pe, inputs)

    for order in [1, 2, 3]:
        if args.alg_name in ["ALL", f"mecdyna{order}_pe"]:
            for i in range(len(alpha_indices)):
                alpha_index = alpha_indices[i]
                alpha = alpha_vals[alpha_index]
                mecdyna_out_dir = setup_alg_output_dir(config, "exp_pe_sample", f"mecdyna{order}_pe", smoothing_param=alpha, exp_dir=args.exp_dir)
            with Pool(24) as p:

                inputs = []
                for trial in range(args.first_trial, args.first_trial + num_trials):
                    inputs.append({"order": order,
                                    "mdp": mdp,
                                    "policy": policy,
                                    "config": config,
                                    "config_path": args.config,
                                    "alpha_vals": alpha_vals,
                                    "alpha_indices": alpha_indices,
                                    "num_iterations": num_iterations,
                                    "trial": trial,
                                    "exp_dir": args.exp_dir,
                                    "model_class": model_class,
                                    })
                p.map(run_mecdyna_pe, inputs)

