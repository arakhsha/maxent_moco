import argparse
import numpy as np
import yaml
import shutil
from algorithms.MECVI_Control import MECVI_Control
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from algorithms.MSVI_Control import MSVI_Control
from algorithms.VI_Control import VI_Control
from utilities import setup_problem, setup_alg_output_dir, get_exp_dir

ROOT_OUTPUT_DIR = "./output"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config', help='Path  of config file')
    parser.add_argument('alg_name', help='Algorithm to run. "ALL" to run all, "None" to just plot')
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    parser.add_argument('--alpha_start', default=1, help='Start k to plot alpha plot')
    parser.add_argument('--alpha_stop', default=15, help='End k to plot alpha plot')
    parser.add_argument('--alpha_step', default=5, help='Interval at which to plot alpha plot')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if config["env"] == "garnet":
        iterations = config["garnet_instances"]
    else:
        iterations = 1

    for i in range(iterations):

        exp_name = "exp_control_vector"
        exp_dir = get_exp_dir(config, "exp_control_vector", exp_dir=args.exp_dir)
        mdp = setup_problem(config, seed=i)

        num_iterations = config["exp_control_exact_num_iterations"]
        alpha_vals = config["exp_control_exact_alphas"]
        # alpha_vals_sensitivity = config["exp_control_exact_alphas_sensitivity"]
        # alpha_val_sensitivity = list(np.around(np.arange(0, 1, alpha_vals_sensitivity), decimals=2))
        # all_alphas = sorted(list(set(alpha_vals + alpha_val_sensitivity)))
        all_alphas = sorted(list(set(alpha_vals)))

        if config["exp_control_exact_model_type"] == "LocallySmoothed":
            model_class = LocallySmoothedModel
        elif config["exp_control_exact_model_type"] == "IdentitySmoothed":
            model_class = IdentitySmoothedModel
        else:
            assert False

        # Running Algorithms
        if args.alg_name in ["ALL", "vi_control"]:
            vi_control = VI_Control(mdp)
            vi_out_dir = setup_alg_output_dir(config, "exp_control_vector{}".format(i), "vi_control",
                                                smoothing_param=None, exp_dir=args.exp_dir)
            vi_control.run(num_iterations, f"{vi_out_dir}/policy_trace.npy", f"{vi_out_dir}/V_trace.npy")
            shutil.copyfile(src=args.config, dst=f"{vi_out_dir}/config.yaml")


        if args.alg_name in ["ALL", "msvi_control"]:
            for alpha in all_alphas:
                Phat = model_class.get_P_hat_using_P(mdp.P(), alpha)
                msvi_control = MSVI_Control(mdp, Phat)
                msvi_out_dir = setup_alg_output_dir(config, "exp_control_vector{}".format(i), "msvi_control",
                                                    smoothing_param=alpha, exp_dir=args.exp_dir)
                msvi_control.run(num_iterations, f"{msvi_out_dir}/policy_trace.npy", f"{msvi_out_dir}/V_trace.npy")
                shutil.copyfile(src=args.config, dst=f"{msvi_out_dir}/config.yaml")

        for order in [1, 2, 3]:
            if args.alg_name in ["ALL", f"mecdyna{order}_control"]:
                for alpha in all_alphas:
                    Phat = model_class.get_P_hat_using_P(mdp.P(), alpha)
                    mecvi_control = MECVI_Control(mdp, Phat, order)
                    mecvi_out_dir = setup_alg_output_dir(config, "exp_control_vector{}".format(i), f"mecvi{order}_control",
                                                        smoothing_param=alpha, exp_dir=args.exp_dir)
                    mecvi_control.run(num_iterations, f"{mecvi_out_dir}/policy_trace.npy", f"{mecvi_out_dir}/V_trace.npy")
                    shutil.copyfile(src=args.config, dst=f"{mecvi_out_dir}/config.yaml")
