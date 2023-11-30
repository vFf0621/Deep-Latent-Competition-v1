import torch
import torch.nn as nn
from torch.distributions import TanhTransform

from dreamer.utils.utils import create_normal_dist, build_network


class Actor(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.actor
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size


        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            action_size*2,
        )

    def forward(self, posterior, deterministic):
        x = torch.cat((posterior.squeeze(0), deterministic.squeeze(0)), -1)
        x = self.network(x)
        dist = create_normal_dist(
                x,
                None,
                min_std=self.config.min_std,
                activation=torch.tanh,
        )
        action = dist.rsample()
        log_probs = dist.log_prob(action)
        log_probs -= torch.log(1-torch.tanh(action).pow(2)+10e-8)
        log_probs = log_probs.sum(-1).unsqueeze(-1)

        return action, log_probs