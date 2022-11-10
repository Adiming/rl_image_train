from stable_baselines3 import DDPG 
from stable_baselines3.common.env_checker import check_env

from env import PlaceEnv
from typing import Callable
import os

model_name="DDPG_test_1e6_gamma09"
dir = os.getcwd()

file_name = os.path.join(dir,"models",model_name)

def keep_train():
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func
    
    model = DDPG.load(file_name)

    check_env(PlaceEnv())
    env = PlaceEnv()

    model.set_env(env)

    model.learn(total_timesteps=int(5e5),
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name=file_name
    )

    model.save(file_name+"_c5e5")

if __name__ == '__main__':
    keep_train()