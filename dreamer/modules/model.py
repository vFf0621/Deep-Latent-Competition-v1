import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from dreamer.utils.utils import create_normal_dist, build_network, horizontal_forward


class RSSM(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm

        self.recurrent_model = RecurrentModel(action_size, config)
        self.transition_model = TransitionModel(config)
        self.representation_model = RepresentationModel(config)

    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(
            batch_size
        ), self.recurrent_model.input_init(batch_size)


class RecurrentModel(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.bs = config.parameters.dreamer.batch_size
        self.bl = config.parameters.dreamer.batch_length
        self.config = config.parameters.dreamer.rssm.recurrent_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.prev_bs = 0
        self.activation = getattr(nn, self.config.activation)()
        self.linear = nn.Linear(
            self.stochastic_size + action_size, self.config.hidden_size
        )
        self.cur_cx = None
        self.cur_hx = None
        self.recurrent = nn.LSTMCell(self.config.hidden_size, self.deterministic_size)
        self.hx = torch.zeros(self.deterministic_size).to(self.device)
        self.cx = torch.zeros(self.deterministic_size).to(self.device)

    def forward(self, embedded_state, action):
        if action is None or embedded_state is None:
            return
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(action).to(self.hx.device)
        x = torch.cat((embedded_state.squeeze(0), action.squeeze(0)),-1)
        x = self.activation(self.linear(x)).squeeze(0)
        if self.cur_hx is not None:
            self.cur_hx, self.cur_cx = self.recurrent(x, (self.cur_hx.squeeze(0), self.cur_cx.squeeze(0)))
        else:
            self.cur_hx, self.cur_cx = self.recurrent(x, (self.hx.squeeze(0), self.cx.squeeze(0)))
        return self.cur_hx

    def input_init(self, batch_size):
        self.cur_cx = None
        self.cur_hx = None
        if batch_size and self.prev_bs == 0:
            self.hx = self.hx.repeat(batch_size, 1)
            self.cx = torch.zeros_like(self.hx).to(self.hx.device)

            self.prev_bs = batch_size
            
        elif batch_size and self.prev_bs != 0:
            self.hx = self.hx[0].repeat(batch_size,1 )
            if batch_size == self.bs * (self.bl-1):
                self.hx = self.hx.detach()
            self.cx = torch.zeros_like(self.hx).to(self.hx.device)

            self.prev_bs = batch_size
        elif self.prev_bs > 0 and batch_size == 1:
            self.hx = self.hx[0].detach()
            self.cx = torch.zeros_like(self.hx).to(self.hx.device)

            self.prev_bs = 0
        return self.hx

class TransitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.transition_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.stochastic_size * 2,
        )

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.config.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).to(self.device)


class RepresentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.representation_model
        self.embedded_state_size = config.parameters.dreamer.embedded_state_size
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.embedded_state_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.stochastic_size * 2,
        )

    def forward(self, embedded_observation, deterministic):
        if embedded_observation is None or deterministic is None:
            return

        x = self.network(torch.cat((embedded_observation, deterministic.squeeze(0)), -1))
        posterior_dist = create_normal_dist(x, min_std=self.config.min_std)
        posterior = posterior_dist.rsample()

        return posterior_dist, posterior


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.reward
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class ContinueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.continue_
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = torch.distributions.Bernoulli(logits=x)
        return dist
