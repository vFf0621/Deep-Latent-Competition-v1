import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import  make_env, get_env_infos
from dreamer.envs.simulate import simulate
import gymnasium as gym
import gym_multi_car_racing

def main(config_file):
    config = load_config(config_file)

    env = gym.make("MultiCarRacing-v1", num_agents = 2)    
    obs_shape=(3, 96, 96)
    discrete_action_bool = False
    action_size = 2

    log_dir = (
        get_base_directory()
        + "/runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    writer = SummaryWriter(log_dir)
    device = config.operation.device

    agents = []
    for i in range(env.num_agents):
        agent = Dreamer(i,obs_shape, discrete_action_bool, action_size, writer, device, config)
        agents.append(agent)

    simulate(agents, env, writer=writer,num_interaction_episodes=5000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="multi_car.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    main(parser.parse_args().config)
