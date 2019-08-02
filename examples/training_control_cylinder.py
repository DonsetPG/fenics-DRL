print('Importing modules...')


# -------

import logging
import os
import sys
from datetime import datetime

import numpy                                as np
import tensorflow                           as tf
from control_cylinder import FluidMechanicsEnv_
from stable_baselines import PPO2, A2C, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

# -------
# -------

# -------

# --------------------------------------------------------------


# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------- Work in Progress -----------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------



print('Modules imported')

# ---------------------------------------


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, console_log_dir, model_file_name, log_name

    # This is invoked in every update
    if (n_steps + 1) % 1 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(console_log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps', file=sys.stderr)
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward),
                file=sys.stderr)

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model

    n_steps += 1
    return True

# ---------------------------------------

logger = logging.getLogger(__name__)

# Create log dir
log_dir = "log/"
# this enables multiple experiments to be run in parallel --> there is one monitor file for each experiment
console_log_dir = log_dir + "console/" + 'obstacles_simple_1' + "/"
models_log_dir = log_dir + "models/"
tensorboard_log_dir = log_dir + "tensorboard/"
img_log_dir = log_dir + "img/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(console_log_dir, exist_ok=True)
os.makedirs(models_log_dir, exist_ok=True)
os.makedirs(img_log_dir, exist_ok=True)


# ---------------------------------------

def launch_training(nb_cpu,name_agent,name_env,total_timesteps,text):

    env_name = name_env
    #n_cpu = 8
    n_cpu = nb_cpu

    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32,32])
    print('TB available at := ',tensorboard_log_dir, file=sys.stderr)



    if name_agent =='A2C':
        env_ = FluidMechanicsEnv_()
        env_ = Monitor(env_, console_log_dir,allow_early_resets=True)

        env = SubprocVecEnv([lambda: env_ for i in range(n_cpu)])
        model = A2C(MlpPolicy, env, n_steps=20,gamma = 0.9, verbose=1,tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
        #model = A2C.load("first_test")
        model_name = "A2C_default_Mlp"+text
    elif name_agent == 'PPO2':
        env_ = FluidMechanicsEnv_()
        env_ = Monitor(env_, console_log_dir,allow_early_resets=True)

        env = SubprocVecEnv([lambda: env_ for i in range(n_cpu)])
        model = PPO2(MlpPolicy, env,n_steps=20,gamma = 1.0, verbose=1,tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
        #model = A2C.load("first_test")
        model_name = "PPO2_default_Mlp"+text
    elif name_agent == 'TRPO':
        env_ = FluidMechanicsEnv_()
        env_ = Monitor(env_, console_log_dir,allow_early_resets=True)

        env = DummyVecEnv([lambda: env_ for i in range(n_cpu)])

        model = TRPO(MlpPolicy, env,gamma = 0.1, verbose=1,tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
        #model = A2C.load("first_test")
        model_name = "TRPO_default_Mlp"+text


    time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

    log_name = f"_model={model_name}_time={time}"
    print('with the following line := ','tensorboard --logdir ',tensorboard_log_dir+log_name)
    training_log = open(f"{console_log_dir}/{log_name}.log", "a")
    sys.stdout = training_log
    logging.basicConfig(level=logging.INFO, filename=f"{console_log_dir}/{log_name}.log", datefmt='%H:%M:%S',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    model_file_name = f"{models_log_dir}{log_name}_best.pkl"


    start = datetime.now()
    print("Learning model", file=sys.stderr)

    model.learn(total_timesteps=int(total_timesteps), tb_log_name=log_name, callback=callback)

    training_time = datetime.now() - start
    print(f"Training time: {training_time}", file=sys.stderr)

    print("Saving final model", file=sys.stderr)
    model.save(f"{models_log_dir}{log_name}_final.pkl")


# ---------------------------------------
# ---------------------------------------
# ---------------------------------------
# ---------------------------------------


