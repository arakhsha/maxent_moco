import argparse
import time
import numpy as np
import yaml
import shutil
import random
from algorithms.MECDyna_Control import MECDyna_Control
from model.CliffWalkAdvModel import CliffWalkAdvModel
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from multiprocessing import Pool
from algorithms.QLearning import QLearning
from algorithms.MSDyna_Control import MSDyna_Control
from algorithms.Dyna_Control import Dyna_Control
from LearningRate import LearningRate
from utilities import setup_problem, setup_alg_output_dir, get_exp_dir, get_default_alg_output_dir

ROOT_OUTPUT_DIR = "./output"

def run_qlearning(inputs):
    mdp, config, config_path, num_iterations, trial, exp_dir, model_class = inputs["mdp"], inputs["config"], inputs["config_path"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    qlearning_pe = QLearning(mdp)
    qlearning_out_dir = get_default_alg_output_dir(config, "exp_control_sample", "qlearning_control", smoothing_param=None,
                                             exp_dir=exp_dir)
    print(qlearning_out_dir)
    lr_scheduler = LearningRate(config["exp_control_lr_type"],
                                config["exp_control_qlearning_lr"],
                                config["exp_control_qlearning_delay"],
                                config["exp_control_qlearning_gamma"])

    shutil.copyfile(src=config_path, dst=f"{qlearning_out_dir}/config.yaml")
    qlearning_pe.run(num_iterations,
                     lr_scheduler=lr_scheduler,
                     policy_filename=f"{qlearning_out_dir}/policy_trace.npy",
                     value_filename=f"{qlearning_out_dir}/value_trace_{trial}.npy")


def run_dyna_control(inputs):
    mdp, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        dyna_control = Dyna_Control(mdp, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        dyna_out_dir = get_default_alg_output_dir(config, "exp_control_sample", "dyna_control", smoothing_param=alpha,
                                            exp_dir=exp_dir)
        print(dyna_out_dir)
        planning_interval = config["exp_control_dyna_planning_interval"][alpha_index]
        shutil.copyfile(src=config_path, dst=f"{dyna_out_dir}/config.yaml")
        dyna_control.run(num_iterations, planning_interval, policy_filename=f"{dyna_out_dir}/policy_trace_{trial}.npy",
                         value_filename=f"{dyna_out_dir}/value_trace_{trial}.npy")


def run_msdyna_control(inputs):
    mdp, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        msdyna_control = MSDyna_Control(mdp, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        msdyna_out_dir = get_default_alg_output_dir(config, "exp_control_sample", "msdyna_control", smoothing_param=alpha,
                                            exp_dir=exp_dir)
        print(msdyna_out_dir)
        lr_scheduler = LearningRate(config["exp_control_lr_type"],
                                    config["exp_control_msdyna_lr"][alpha_index],
                                    config["exp_control_msdyna_delays"][alpha_index],
                                    config["exp_control_msdyna_gamma"][alpha_index])
        shutil.copyfile(src=config_path, dst=f"{msdyna_out_dir}/config.yaml")
        msdyna_control.run(num_iterations,
                           lr_scheduler=lr_scheduler,
                           policy_filename=f"{msdyna_out_dir}/policy_trace_{trial}.npy",
                           value_filename=f"{msdyna_out_dir}/value_trace_{trial}.npy")

def run_mecdyna_control(inputs):
    order, mdp, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  \
        = inputs["order"], inputs["mdp"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        beta = config[f"exp_control_mecdyna{order}_beta"][alpha_index]
        planning_interval = config[f"exp_control_mecdyna{order}_planning_interval"][alpha_index]
        val_queue_interval = config[f"exp_control_mecdyna{order}_val_queue_interval"][alpha_index]
        num_unused_V = config[f"exp_control_mecdyna{order}_num_unused_V"][alpha_index]

        mecdyna_control = MECDyna_Control(order, mdp, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        mecdyna_out_dir = get_default_alg_output_dir(config, "exp_control_sample", f"mecdyna{order}_control", smoothing_param=alpha,
                                            exp_dir=exp_dir)
        print(mecdyna_out_dir)
        shutil.copyfile(src=config_path, dst=f"{mecdyna_out_dir}/config.yaml")
        mecdyna_control.run(num_iterations,
                           num_unused_V,
                           beta,
                           planning_interval,
                           val_queue_interval,
                           policy_filename=f"{mecdyna_out_dir}/policy_trace_{trial}.npy",
                           value_filename=f"{mecdyna_out_dir}/value_trace_{trial}.npy",
                           uncorrected_model_filename=f"{mecdyna_out_dir}/phat_trace_{trial}.npy",
                           corrected_model_filename=f"{mecdyna_out_dir}/pbar_trace_{trial}.npy",
                           opt_time_filename=f"{mecdyna_out_dir}/opt_time_trace_{trial}.npy",
                           iter_filename=f"{mecdyna_out_dir}/iter_trace_{trial}.npy")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config', help='Path of config file')
    parser.add_argument('alg_name', help='Algorithm to run. "ALL" to run all, "None" to just plot')
    parser.add_argument('--num_trials', default=1, help='Number of trials to run')
    parser.add_argument('--alpha_index', default=-1, type=int, help='Index of alpha value to run. -1 for all.')
    parser.add_argument('--first_trial', default=0, type=int, help='Index of alpha value to run. -1 for all.')
    parser.add_argument('--parallel', action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_control_sample"
    exp_dir = get_exp_dir(config, "exp_control_sample", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    num_iterations = config["exp_control_sample_num_iterations"]
    alpha_vals = config["exp_control_sample_alphas"]
    if args.alpha_index == -1:
        alpha_indices = [i for i in range(len(alpha_vals))]
    else:
        alpha_indices = [args.alpha_index]

    num_trials = int(args.num_trials)

    if config["exp_control_sample_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_control_sample_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    elif config["exp_control_sample_model_type"] == "CliffWalkAdvSmoothed":
        model_class = CliffWalkAdvModel
    else:
        assert False

    run_times = np.zeros((3, 6))

    # Running Algorithms
    if args.alg_name in ["ALL", "qlearning"]:
        tdlearning_out_dir = setup_alg_output_dir(config, "exp_control_sample", "qlearning_control", smoothing_param=None,
                                                  exp_dir=args.exp_dir)
        with Pool(24) as p:
            inputs = []
            start_time = time.time()
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                               "config": config,
                               "config_path": args.config,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
                if not args.parallel:
                    run_qlearning(inputs[trial])
            if args.parallel:
                p.map(run_qlearning, inputs)
            else:
                run_times[:, 0] = (time.time() - start_time) / num_trials

    if args.alg_name in ["ALL", "dyna_control"]:
        for i in range(len(alpha_indices)):
            alpha_index = alpha_indices[i]
            alpha = alpha_vals[alpha_index]
            dyna_out_dir = setup_alg_output_dir(config, "exp_control_sample", "dyna_control", smoothing_param=alpha,
                                                exp_dir=args.exp_dir)
        with Pool(24) as p:
            inputs = []
            start_time = time.time()
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "alpha_indices": alpha_indices,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
                if not args.parallel:
                    run_dyna_control(inputs[trial])
            if args.parallel:
                p.map(run_dyna_control, inputs)
            else:
                run_times[:, 1] = (time.time() - start_time) / num_trials

    if args.alg_name in ["ALL", "msdyna_control"]:
        for i in range(len(alpha_indices)):
            alpha_index = alpha_indices[i]
            alpha = alpha_vals[alpha_index]
            msdyna_out_dir = setup_alg_output_dir(config, "exp_control_sample", "msdyna_control", smoothing_param=alpha, exp_dir=args.exp_dir)

        with Pool(24) as p:

            inputs = []
            start_time = time.time()
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "alpha_indices": alpha_indices,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
                if not args.parallel:
                    run_msdyna_control(inputs[trial])
            if args.parallel:
                p.map(run_msdyna_control, inputs)
            else:
                run_times[:, 2] = (time.time() - start_time) / num_trials

    for order in [1, 2, 3]:
        if args.alg_name in ["ALL", f"mecdyna{order}_control", "mecdyna_control"]:
            for i in range(len(alpha_indices)):
                alpha_index = alpha_indices[i]
                alpha = alpha_vals[alpha_index]
                mecdyna_out_dir = setup_alg_output_dir(config, "exp_control_sample", f"mecdyna{order}_control", smoothing_param=alpha, exp_dir=args.exp_dir)
            with Pool(24) as p:

                inputs = []
                start_time = time.time()
                for trial in range(args.first_trial, args.first_trial + num_trials):
                    inputs.append({"order": order,
                                    "mdp": mdp,
                                    "config": config,
                                    "config_path": args.config,
                                    "alpha_vals": alpha_vals,
                                    "alpha_indices": alpha_indices,
                                    "num_iterations": num_iterations,
                                    "trial": trial,
                                    "exp_dir": args.exp_dir,
                                    "model_class": model_class,
                                    })
                    if not args.parallel:
                        run_mecdyna_control(inputs[trial])
                if args.parallel:
                    p.map(run_mecdyna_control, inputs)
                else:
                    run_times[:, 2 + order] = (time.time() - start_time) / num_trials

    print(run_times)

