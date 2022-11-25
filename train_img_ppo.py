from gym import Env
from stable_baselines3 import DDPG, TD3, DQN,PPO
import torch
from env_image_ppo import PlaceEnvImage
from typing import Callable
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.ddpg import CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
#from features import CustomCNN, CustomCNNBN, CustomCNNRes, CustomCNN128, CustomCNNLSTM
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
import csv
import time
import os

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # n_input_channels = observation_space.shape[0]
        n_input_channels = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn2(self.cnn(th.as_tensor(observation_space.sample()[None]).float())).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn2(self.cnn(observations)))

feature_mapping = {
    'CustomCNN': CustomCNN
}

file_name="ppo_img_1e6_ppo"

class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals["total_timesteps"])

    def _on_step(self):
        self.progress_bar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


def get_make_env(
    check: bool = True
) -> Callable[[], Env]:
    def _init():
        env = PlaceEnvImage()        
        check and env.check()
        return Monitor(env, filename=file_name)

    return _init


def create_envs(
    num_training_envs: int = 1, num_eval_envs: int = 1
):
    make_env = get_make_env(check=False)

    train_env = (
        DummyVecEnv([make_env])
        if num_training_envs == 1
        else SubprocVecEnv([make_env] * num_training_envs)
    )
    eval_env = (
        DummyVecEnv([make_env])
        if num_eval_envs == 1
        else SubprocVecEnv([make_env] * num_eval_envs)
    )

    return train_env, eval_env


def img_train(
    train_env: Env,
    eval_env: Env,
    num_eval_points: int = 20,
    feature = 'CustomCNN',
    save_dir='models',
    train_steps=10_00000,
    gamma=0.9,
    learning_rate=1e-3
):

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value
        return func

    # The noise objects 
    # n_actions = train_env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = PPO(
        'CnnPolicy',
        train_env,
        # buffer_size=100000,
        batch_size=128,
        gamma=gamma,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1,
        tensorboard_log= './logs',
        learning_rate=linear_schedule(learning_rate),
        # action_noise = action_noise,
        policy_kwargs=dict(
            features_extractor_class=feature_mapping[feature],
            features_extractor_kwargs={},
            ),
        # gradient_steps=-1,
        # train_freq=(1, "step"),
    )

    # Creating callbacks for eval (reward over time) and checkpoints (saving the model in spec. frequency)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path="./logs/")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=train_steps // train_env.num_envs // num_eval_points,
        n_eval_episodes=1,
    )
    callback = CallbackList([TqdmCallback(), checkpoint_callback, eval_callback])

    model.learn(total_timesteps=train_steps,
                # progress_bar=True,
                log_interval=100,
                tb_log_name=file_name,
                callback=callback,
                reset_num_timesteps=False
    )

    model.save(f'{save_dir}/{file_name}_{train_steps}')
    print("saved model successfully")

def predict_and_write():
    header = ["i","r","x","y","gx","gy"]

    env = PlaceEnvImage()

    dir = os.getcwd()
    # dir = os.path.join(dir,"models") 
    dir = os.path.join(dir,"logs") 
    # f_name = file_name + "_10000"
    f_name = file_name 
    f_name = "rl_model_400000_steps"
    file = os.path.join(dir,f_name)
    model = PPO.load(file,env=env)
    
    for i in range(1):
        obs = env.reset()
        # print(type(obs))
        dones = False
        total_reward = 0
        with open(file_name+'.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            while True:
                action,_ = model.predict(obs,deterministic=True)                
                # print(type(action)) #numpy.ndarray
                obs , reward, dones, info = env.step(action)
                total_reward += reward

                # log coordinate
                data = [info['i'],reward,info['x'],info['y'],info['g_x'],info['g_y']]
                writer.writerow(data)

                env.render()
                time.sleep(1)
                # print("step:{},step_r:{}, total_torque:{}, old_torque:{}, init_torque:{}".format(info['i'],rewards,info['t_s'],info['o_s'],info['i_s']))
                print("step:{},step_r:{},x:{},y:{},gx:{},gy:{}".format(info['i'],reward,info['x'],info['y'],info['g_x'],info['g_y']))
                if dones:
                    print(total_reward)
                    break


if __name__ == '__main__':
    envs = create_envs(num_training_envs=1, num_eval_envs=1)    # the num can up to 16, if possible keep eval and training env same
    img_train(*envs)

    # predict_and_write()