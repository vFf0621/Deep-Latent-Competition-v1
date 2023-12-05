import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
import wandb
from dreamer.algorithms.dreamerv3LSTM import DreamerV3
from dreamer.utils.utils import load_config, get_base_directory
from simulate import simulate
import gymnasium as gym
import gym_multi_car_racing

'''
The main file initializes the multiCarRacing-v1 gym environment and uses the configuration
from ../configs file to initialize the agents. The wandb is used to put the data into graphs. 

The environment is found in the multi_car_racing folder

One of the cars uses an LSTM (red car) as the recurrent model and the other car uses a GRU (blue car)

'''


def main(config_file1, config_file2):
    config1 = load_config(config_file1+".yml")
    config2 = load_config(config_file2+".yml")
    env = gym.make("MultiCarRacing-v1", num_agents = 2)  
    obs_shape=env.observation_space.shape
    action_size = 2
    config = config1
    log_dir = (
        get_base_directory()
        + "/runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    project_name = 'multi_car ' + config.algorithm

    with wandb.init(project=project_name, entity='fguan06', config=dict(config), settings=wandb.Settings(start_method="thread")):

        device = config.operation.device

        agents = []
        for i in range(env.num_agents):
            if i % 2 == 0:
                if config1.algorithm == "dreamer-v3":
                    agent = DreamerV3(i,obs_shape, action_size, dict(), device, config2, LSTM=0)
                else:
                    agent = DreamerV3(i,obs_shape, action_size, dict(), device, config2,LSTM=1)
                if config1.parameters.load:
                    agent.load_state_dict()

            else:
                if config2.algorithm == "dreamer-v3":
                    agent = DreamerV3(i,obs_shape, action_size, dict(), device, config1, LSTM=0)
                else:
                    agent = DreamerV3(i,obs_shape, action_size, dict(), device, config1, LSTM=1)
                if config2.parameters.load:
                    agent.load_state_dict()

            agents.append(agent)

        simulate(agents, env, writer=dict(),num_interaction_episodes=903)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent1",
        type=str,
        default="Dreamerv3LSTM",
        help="Algorithm to run on odd number agents (Default=Dreamerv3LSTM)",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default="Dreamerv3",
        help="Algorithm to run on even number agents (Default=Dreamerv3)",
    )
    print()
    main(parser.parse_args().agent1, parser.parse_args().agent2)
