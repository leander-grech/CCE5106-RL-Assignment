import argparse
from copy import deepcopy
import os
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from cce5106.envs.um_flight_env import UMFlightEnv
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on UMFlightEnv")
    parser.add_argument('model_path', type=str, help='Path to the trained PPO model')
    parser.add_argument('--n_episodes', type=int, default=1, help='Number of episodes to run for evaluation')
    # parser.add_argument('--env_kwargs', type=dict, default={}, help='Environment kwargs')
    return parser.parse_args()


def main(args):
    # Load the trained model
    model_path = args.model_path
    model_name = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(model_path), '..')))
    # model_path = '/home/leander/code/ASTRA/astra/agents/PPO_UMFlightEnv/07012024-150853_UMFlightEnv/models/ppo_umflightenv_1000000_steps.zip'
    N = 5
    M = 10
    max_steps = 50

    model = PPO.load(model_path)

    env = UMFlightEnv(n=N, m=M, max_steps=max_steps)

    z_action = np.zeros(shape=env.action_space.shape)
    zenv = UMFlightEnv(n=N, m=M, max_steps=max_steps)

    # Create renders directory if it doesn't exist
    renders_dir = os.path.join('experiments', model_name)
    os.makedirs(renders_dir, exist_ok=True)
    print(f'Saving renders to {renders_dir}')

    # Evaluate the model
    for episode in range(args.n_episodes):
        obs = env.reset()[0]
        zenv.reset()
        zenv.sim = deepcopy(env.sim)
        term, trunc = False, False
        episode_rewards = 0
        rews = []
        while not (term or trunc):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            zenv.step(z_action)
            episode_rewards += reward
            rews.append(reward)

        print(f'Episode {episode + 1}: Total reward: {episode_rewards}')
        # Save the render to the renders directory
        render_path = os.path.join(renders_dir, f'episode_{episode + 1}.mp4')
        zrender_path = os.path.join(renders_dir, f'zero_action_episode_{episode + 1}.mp4')
        env.render(render_path)
        zenv.render(zrender_path)
        # plt.show()
    # env.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
