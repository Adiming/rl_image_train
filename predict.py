from stable_baselines3 import DDPG, PPO
from env import PlaceEnv
import time
import os
import csv
import numpy as np
from typing import Callable
from stable_baselines3.common.env_checker import check_env
import torch as th

file_name="DDPG_5e5_lr1e_nrw_p10_lowp_2"
# file_name="DDPG_test_1e6_ongoal"

# single epochs
def test_train():

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    check_env(PlaceEnv())
    env = PlaceEnv()
    # file_name="DDPG_test_lr2e_3"

    model = DDPG(
        "MlpPolicy",
        env,
        buffer_size=20_0000,
        batch_size=128,
        gamma=0.9,
        verbose=1,
        tensorboard_log= './logs',
        learning_rate=linear_schedule(0.001),
        policy_kwargs=dict(
            # pi:for the actor; qf:for the Q function
            net_arch=dict(pi=[256,256],qf=[256,256]))
    )

    model.learn(total_timesteps=int(5e5),
                progress_bar=True,
                tb_log_name=file_name
    )
    model.save(file_name)
    del model

    env.close()

def test_train_PPO():

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    check_env(PlaceEnv())
    env = PlaceEnv()
    # file_name="DDPG_test_lr2e_3"

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log= './logs',
        learning_rate=linear_schedule(0.002),
        policy_kwargs=dict(
            # pi:for the actor; qf:for the Q function
            # tactivation_fn = th.nn.ReLU,
            net_arch=[dict(pi=[256,256],vf=[256,256])])
    )

    model.learn(total_timesteps=int(1e6),
                progress_bar=True,
                tb_log_name=file_name
    )
    model.save(file_name)
    del model

    env.close()


def predict_and_write():
    header = ["i","r","x","y","d"]

    env = PlaceEnv()

    dir = os.getcwd()
    # dir = os.path.join(dir,"models","1_100000_513") 
    # dir = os.path.join(dir,"models_2","30_3000000_-681") 
    dir = os.path.join(dir,file_name) 

    model = DDPG.load(dir,env=env)
    # model = PPO.load(dir,env=env)

    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(mean_reward)
    
    for i in range(1):
        obs = env.reset()
        dones = False
        total_reward = 0
        with open(file_name+'.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            while True:
                action,_ = model.predict(obs,deterministic=True)                
                # print(action)
                obs , rewards, dones, info = env.step(action)
                total_reward += rewards

                # log coordinate
                data = [info['i'],rewards,info['x'],info['y'],info['distance']]
                writer.writerow(data)

                env.render()
                time.sleep(1)
                # print("step:{},step_r:{}, total_torque:{}, old_torque:{}, init_torque:{}".format(info['i'],rewards,info['t_s'],info['o_s'],info['i_s']))
                print("step:{},step_r:{},distance:{},init_torque:{},old_torque:{}".format(info['i'],rewards,info['distance'],info['i_s'],info['o_s']))
                if dones:
                    print(total_reward)
                    break

if __name__ == '__main__':
    # test_train_PPO()
    # test_train()
    predict_and_write()
