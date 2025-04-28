import argparse
import os
import json
from torch.nn import ReLU
from datetime import datetime as dt
from stable_baselines3 import PPO
# TODO: Uncomment this to use SB3 callbacks
# from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from cce5106.envs.um_flight_env import UMFlightEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on UMFlightEnv")
    # UMFLightEnv parameters
    parser.add_argument('--n', type=int, default=5, help='Number of flights initialised on a collision course')
    parser.add_argument('--m', type=int, default=5, help='Number of flights initialised on a random course')
    parser.add_argument('--max-steps', type=int, default=50, help='Maximum number of time-steps per episode')
    # TODO: Tweak reward coefficients
    parser.add_argument('--collision-coeff', type=float, default=1.0, help='Collision coefficient scale used in reward method')
    parser.add_argument('--clearance-coeff', type=float, default=1.0, help='Number of clearances coefficient scale used in reward method')
    parser.add_argument('--invalid-coeff', type=float, default=1.0, help='Invalid actions coefficient scale used in reward method')

    # RL algorithm parameters
    parser.add_argument('--n-envs', type=int, default=16, help='Number of training vectorised environments')
    parser.add_argument('--algo', type=str, default='PPO', help='Type of RL agent')
    parser.add_argument('--n-steps', type=int, default=200, help='The number of steps to run for each environment per update'
                                                               '(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)'
                                                               'NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)'
                                                               'See https://github.com/pytorch/pytorch/issues/29372')
    parser.add_argument('--n-epochs', type=int, default=16, help='Number of epoch when optimizing the surrogate loss')
    parser.add_argument('--retrain-latest', action='store_true')

    # Training session parameters
    parser.add_argument('--train-steps', type=int, default=1500000, help='Number of training steps')
    parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of episodes for evaluation')
    parser.add_argument('--eval-freq', type=int, default=500, help='Frequency of evaluations in steps')
    parser.add_argument('--log-freq', type=int, default=100, help='Frequency of logging in steps')
    parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of checkpoints in steps')
    parser.add_argument('--seed', default=123, type=int, help='Set random seed')

    return parser.parse_args()


def ppo_learning_rate(x):
    lr1 = 3e-4
    return lr1 * x


def main(args, SEED=None):
    algo_str = args.algo

    if algo_str == 'PPO':
        algo = PPO
    else:
        raise NotImplementedError

    env_kwargs = dict(n=args.n,
                      m=args.m,
                      max_steps=args.max_steps,
                      rew_coeffs = dict(collision=args.collision_coeff,
                                        clearance=args.clearance_coeff,
                                        invalid=args.invalid_coeff)
    )

    # Create environments
    env = make_vec_env(lambda: UMFlightEnv(**env_kwargs), n_envs=args.n_envs)
    eval_env = UMFlightEnv(**env_kwargs)

    model_dir = f'{algo_str}_{str(eval_env)}'
    model_name = dt.now().strftime(f'%m%d%Y-%H%M%S')
    model_path = os.path.join(model_dir, model_name)
    tb_log_dir = os.path.join(model_dir, model_name, 'logs')
    eval_dir = os.path.join(model_dir, model_name, 'evals')
    save_dir = os.path.join(model_dir, model_name, 'models')

    for d in (model_dir, tb_log_dir, eval_dir, save_dir):
        if not os.path.exists(d):
            os.makedirs(d)

    policy_kwargs = dict(activation_fn=ReLU,
                         net_arch=dict(pi=[256, 256], vf=[256, 256]))
    algo_kwargs = dict(learning_rate=ppo_learning_rate, seed=SEED)
    if args.n_steps > 0:
        algo_kwargs['n_steps'] = args.n_steps
    if args.n_epochs > 0:
        algo_kwargs['n_epochs'] = args.n_epochs
    print(eval_env.INVALID_COEFF_SCALE)
    info_fn = 'info.txt'
    TRAINING_MESSAGE = (f"\n{repr(eval_env)}\n"
                        f"Collisions coeff. scale: {eval_env.CLEARANCE_COEFF_SCALE:.2f} (α)\n"
                        f"Clearances coeff. scale: {eval_env.CLEARANCE_COEFF_SCALE:.2f} (β)\n"
                        f"Invalids coeff scale: {eval_env.INVALID_COEFF_SCALE:.2f} (γ)\n"
                        f"Invalids/MaxClearances ratio: {eval_env.INVALID_MAXCLEARANCE_RATIO:.2f} \n")
    print(TRAINING_MESSAGE)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(os.path.join(model_path, info_fn), 'w') as f:
        f.write(USER_MESSAGE)
        f.write(TRAINING_MESSAGE)
        f.write('\nscript args:\n')
        f.write(f'\n{str(args)}\n')
        f.write(f'\nRL algo:  {algo_str}\n')
        f.write('\npolicy_kwargs:\n')
        pk = policy_kwargs.copy()
        pk.pop('activation_fn')
        json.dump(pk, f, indent=10)
        f.write('\nalgo_kwargs:\n')
        ak = algo_kwargs.copy()
        ak.pop('learning_rate')
        json.dump(ak, f, indent=10)

    # TODO: Add training checkpoint and evaluation callbacks here (See: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
    callbacks = []

    # Create the PPO agent
    model = algo('MlpPolicy', env, verbose=1, tensorboard_log=tb_log_dir, policy_kwargs=policy_kwargs, **algo_kwargs)

    # Train the agent
    # TODO: Add tensorboard logging (you can use tb_log_name param and/or explicitly log through custom EvalCallback)
    model.learn(total_timesteps=args.train_steps,
                callback=callbacks,
                log_interval=args.log_freq)

    # Save the final model
    model.save(os.path.join(save_dir, 'ppo_umflightenv_final'))


if __name__ == '__main__':
    args = parse_args()
    USER_MESSAGE = input('You can say something about this training session OR press enter\n')
    main(args)
