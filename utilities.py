import hashlib
import os
import numpy as np

from model.LocallySmoothedModel import LocallySmoothedModel

from env.Garnet import Garnet
from env.Maze import Maze33
from env.CliffWalk import CliffWalk

ROOT_OUTPUT_DIR = "./output"

def get_model_class(model_type):
    if model_type == "LocallySmoothed":
        return LocallySmoothedModel
    assert False

def get_config_name(config):
    id = hashlib.md5(str(config).encode()).hexdigest()[:4]
    env = config["env"]
    return f'{id}_{env}'

def get_default_alg_output_dir(config, exp_name, alg_name, smoothing_param, exp_dir=None):
    if exp_dir is None:
        config_name = get_config_name(config)
        exp_dir = f"{config_name}/{exp_name}"

    if smoothing_param is not None:
        default_exp_dir = f'{alg_name}_{smoothing_param}'
    else:
        default_exp_dir = f'{alg_name}'
    return f"{ROOT_OUTPUT_DIR}/{exp_dir}/{default_exp_dir}"

def get_exp_dir(config, exp_name, exp_dir=None):
    if exp_dir is None:
        config_name = get_config_name(config)
        exp_dir = f"{config_name}/{exp_name}"
    return exp_dir

def setup_alg_output_dir(config, exp_name, alg_name, smoothing_param, exp_dir=None):
    if exp_dir is not None:
        alg_output_dir = f'{ROOT_OUTPUT_DIR}/{exp_dir}'
        if not os.path.isdir(f'{ROOT_OUTPUT_DIR}/{exp_dir}'):
            os.mkdir(f'{ROOT_OUTPUT_DIR}/{exp_dir}')
    else:
        config_name = get_config_name(config)
        
        if not os.path.isdir(f"{ROOT_OUTPUT_DIR}/{config_name}"):
            os.mkdir(f"{ROOT_OUTPUT_DIR}/{config_name}")
        if not os.path.isdir(f'{ROOT_OUTPUT_DIR}/{config_name}/{exp_name}'):
            os.mkdir(f'{ROOT_OUTPUT_DIR}/{config_name}/{exp_name}')
        exp_dir = f"{config_name}/{exp_name}"
        
    if smoothing_param is not None:
        default_exp_dir = f'{alg_name}_{smoothing_param}'
    else:
        default_exp_dir = f'{alg_name}'

    i = 1
    exp_dir = default_exp_dir
    while os.path.isdir(f"{ROOT_OUTPUT_DIR}/{config_name}/{exp_name}/{exp_dir}"):
        i += 1
        exp_dir = default_exp_dir + f"({i})"

    os.mkdir(f"{ROOT_OUTPUT_DIR}/{config_name}/{exp_name}/{exp_dir}")

    alg_output_dir = f"{ROOT_OUTPUT_DIR}/{config_name}/{exp_name}/{exp_dir}"

    return alg_output_dir

def setup_problem(config, seed=0):
    env = config["env"]
    discount = config["discount"]
    if env == "garnet":
        np.random.seed(seed)
        return Garnet(discount, num_states=config["garnet_problem_num_states"], num_actions=config["garnet_problem_num_actions"], b_P=config["garnet_problem_branching_factor"], b_R=config["garnet_problem_non_zero_rewards"])   
    elif env == "maze33":
        return Maze33(config["maze33_success_prob"], discount)
    elif env == "cliffwalk":
        return CliffWalk(config["cliffwalk_success_prob"], discount)
    else:
        assert False    
