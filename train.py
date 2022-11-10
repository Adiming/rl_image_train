from stable_baselines3 import DDPG 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from env import PlaceEnv
import time
from typing import Callable

# multiple epochs
def train(
    epochs = 30,
    train_steps = 10_0000,
    save_dir = 'models_2',
    gamma=0.9,
    learning_rate=1e-3,
    n_test = 5
):
    check_env(PlaceEnv())
    env = PlaceEnv()

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate= learning_rate,
        buffer_size=20_0000,
        batch_size=128,
        gamma=gamma,
        verbose=1,
        tensorboard_log= './logs',
        policy_kwargs=dict(
            # pi:for the actor; qf:for the Q function
            net_arch=dict(pi=[256,256],qf=[256,256]))
    )

    steps = 0
    for e in range(epochs):
        model.learn(
            total_timesteps=train_steps, 
            reset_num_timesteps=False,
            tb_log_name='DDPG_Platform_256_threshold'
        )
        steps += train_steps

        # NOTE: reset for bellow evaluation
        obs = env.reset()
        total_rewards = 0
        for _ in range(n_test):
            total_reward = 0
            while True:
                # deterministic: Whether or not to return deterministic actions
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                # time.sleep(0.1)
                env.render()
                # total_reward += reward[0]
                total_reward += reward
                if done:
                    obs = env.reset()
                    print(total_reward)
                    total_rewards += total_reward
                    break
        
        total_rewards //= n_test    # get the average total_rewards
        # save every epochs's trained model
        # if not delete the model, the same one is trained
        model.save(f'{save_dir}/{e + 1}_{steps}_{int(total_rewards)}')

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
    file_name="DDPG_test_lr2e_3"

    model = DDPG(
        "MlpPolicy",
        env,
        buffer_size=20_0000,
        batch_size=128,
        gamma=0.9,
        verbose=1,
        tensorboard_log= './logs',
        learning_rate=linear_schedule(0.002),
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

# testing
def predict():
    env = PlaceEnv()
    model = DDPG.load("place_gear",env=env)

    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(mean_reward)

    for i in range(10):
        obs = env.reset()
        dones = False
        total_reward = 0
        while True:
            action,_ = model.predict(obs,deterministic=True)
            # print(action)
            obs , rewards, dones, info = env.step(action)
            total_reward += rewards
            env.render()
            time.sleep(1)
            print("step_r:{}, total_torque:{}, old_torque:{}, init_torque:{}".format(rewards,info['t_s'],info['o_s'],info['i_s']))
            if dones:
                print(total_reward)
                break
            

if __name__ == '__main__':
    # train()
    test_train()
    # predict()