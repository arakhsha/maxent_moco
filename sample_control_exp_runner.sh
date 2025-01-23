# MoCoDyna

python exp_control_sample.py config.yaml mecdyna1_control --num_trials 20 --alpha_index 0
python exp_control_sample.py config.yaml mecdyna2_control --num_trials 20 --alpha_index 0
python exp_control_sample.py config.yaml mecdyna3_control --num_trials 20 --alpha_index 0

python exp_control_sample.py config.yaml mecdyna1_control --num_trials 20 --alpha_index 1
python exp_control_sample.py config.yaml mecdyna2_control --num_trials 20 --alpha_index 1
python exp_control_sample.py config.yaml mecdyna3_control --num_trials 20 --alpha_index 1

python exp_control_sample.py config.yaml mecdyna1_control --num_trials 20 --alpha_index 2
python exp_control_sample.py config.yaml mecdyna2_control --num_trials 20 --alpha_index 2
python exp_control_sample.py config.yaml mecdyna3_control --num_trials 20 --alpha_index 2

# OSDyna
python exp_control_sample.py config.yaml msdyna_control --num_trials 20 --alpha_index 0
python exp_control_sample.py config.yaml msdyna_control --num_trials 20 --alpha_index 1
python exp_control_sample.py config.yaml msdyna_control --num_trials 20 --alpha_index 2

# Dyna
python exp_control_sample.py config.yaml dyna_control --num_trials 20 --alpha_index 0
python exp_control_sample.py config.yaml dyna_control --num_trials 20 --alpha_index 1
python exp_control_sample.py config.yaml dyna_control --num_trials 20 --alpha_index 2

# QLearning
python exp_control_sample.py config.yaml qlearning --num_trials 20

# Plotting
python plotter_control_sample.py