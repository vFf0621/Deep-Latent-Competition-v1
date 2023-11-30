import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
import wandb
from dreamerv3.algorithms.dreamerv3 import DreamerV3
from dreamer.algorithms.dreamerv3LSTM import DreamerLSTM
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import  make_env, get_env_infos
from dreamer.envs.simulate import simulate
import gymnasium as gym
import gym_multi_car_racing

def main(config_file1, config_file2):
    config1 = load_config(config_file1)
    config2 = load_config(config_file2)
    env = gym.make("MultiCarRacing-v1", num_agents = 1)  
    obs_shape=(3, 96, 96)
    discrete_action_bool = False
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
            if i % 1 == 0:
                agent = DreamerV3(i,obs_shape, discrete_action_bool, action_size, dict(), device, config2)

            else:
                agent = DreamerLSTM(i,obs_shape, discrete_action_bool, action_size, dict(), device, config1)

            #agent.load_state_dict()

            agents.append(agent)

        simulate(agents, env, writer=dict(),num_interaction_episodes=903)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent1",
        type=str,
        default="1.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default="2.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    print()
    main(parser.parse_args().agent1, parser.parse_args().agent2)
