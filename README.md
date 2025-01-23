# Maximum Entropy Model Correction in Reinforcement Learning
This is the code for experiments in the paper "[Maximum Entropy Model Correction in Reinforcement Learning](https://arxiv.org/abs/2311.17855)" at ICLR2024. 

**Note:** Some algorithm names and the definition of $\beta$ differ from the paper. We have $\beta_\text{code} = 4/\beta^2_\text{paper}$.

**Acknowledgement**: Some of the code is originally from [here]("https://github.com/awwang10/osvi"), which is developed by Andrew Wang and Amin Rakhsha

## To reproduce all the plots
Simply run the scripts in bash:

```
sh ./vector_exp_runner.sh
sh ./sample_control_exp_runner.sh
sh ./sample_pe_exp_runner.sh
```

## To run a dynamic programming experiment

You can use exp_pe_vector.py or exp_control_vector.py for the PE and control problems along a config file. For example:

```
python ./exp_pe_vector.py <configfilename.yaml> ALL
python ./exp_control_vector.py <configfilename.yaml> ALL
```

## To run an RL experiment

You can use exp_pe_sample.py or exp_control_sample.py for the PE and control problems along a config file. For example:
```
python ./exp_pe_sample.py <configfilename.yaml> ALL --num_trials 20
python ./exp_control_sample.py <configfilename.yaml> ALL --num_trials 20
```

