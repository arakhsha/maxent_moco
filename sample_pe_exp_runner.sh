# MoCoDyna

python exp_pe_sample.py config.yaml mecdyna1_pe --num_trials 20 --alpha_index 0
python exp_pe_sample.py config.yaml mecdyna2_pe --num_trials 20 --alpha_index 0
python exp_pe_sample.py config.yaml mecdyna3_pe --num_trials 20 --alpha_index 0

python exp_pe_sample.py config.yaml mecdyna1_pe --num_trials 20 --alpha_index 1
python exp_pe_sample.py config.yaml mecdyna2_pe --num_trials 20 --alpha_index 1
python exp_pe_sample.py config.yaml mecdyna3_pe --num_trials 20 --alpha_index 1

python exp_pe_sample.py config.yaml mecdyna1_pe --num_trials 20 --alpha_index 2
python exp_pe_sample.py config.yaml mecdyna2_pe --num_trials 20 --alpha_index 2
python exp_pe_sample.py config.yaml mecdyna3_pe --num_trials 20 --alpha_index 2

# OSDyna
python exp_pe_sample.py config.yaml msdyna_pe --num_trials 20 --alpha_index 0
python exp_pe_sample.py config.yaml msdyna_pe --num_trials 20 --alpha_index 1
python exp_pe_sample.py config.yaml msdyna_pe --num_trials 20 --alpha_index 2

# Dyna
python exp_pe_sample.py config.yaml dyna_pe --num_trials 20 --alpha_index 0
python exp_pe_sample.py config.yaml dyna_pe --num_trials 20 --alpha_index 1
python exp_pe_sample.py config.yaml dyna_pe --num_trials 20 --alpha_index 2

# TD Learning
python exp_pe_sample.py config.yaml tdlearning_pe --num_trials 20

# Plotting
python plotter_pe_sample.py