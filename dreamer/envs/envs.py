import gymnasium as gym

from dreamer.envs.wrappers import *


import gym_multi_car_racing


def make_env(task_name, num_agents):
    env = gym.make(task_name, num_agents=num_agents)
    return env


def get_env_infos(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_bool = True
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box2D):
        discrete_action_bool = False
        action_size = env.action_space.shape[0]
    else:
        raise Exception
    return obs_shape, discrete_action_bool, action_size
