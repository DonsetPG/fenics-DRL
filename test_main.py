"""

Experiments One :

- Simple reward
- Simple states

"""

import numpy as np

from examples.training_control_cylinder import launch_training

nb_cpu = 16
agents = ['PPO2','A2C','PPO2','A2C']
name_env = 'FM_control_flow_cylinder-v0'
total_timesteps = 10000
text = '1_step_1_episode_16CPU'

for name_agent in agents:

    best_mean_reward, n_steps = -np.inf, 0
    launch_training(nb_cpu,name_agent,name_env,total_timesteps,text)