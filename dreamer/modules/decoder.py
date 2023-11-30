import torch.nn as nn
import torch
from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
    create_normal_dist,
)


class Decoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Linear(
                self.deterministic_size + self.stochastic_size, 4096,
            ),
            nn.LeakyReLU(),
            nn.LayerNorm(4096),
            nn.Unflatten(dim=1, unflattened_size=(-1, 8, 8)),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=12, stride=4),

            
            )
        self.log_std = nn.Parameter(torch.zeros(1)).to(config.operation.device)
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic, seq=0):
        if seq:
            seq_len = posterior.shape[0]
            batch_size = posterior.shape[1]
            posterior = posterior.reshape(-1, posterior.shape[-1])
            deterministic = deterministic.reshape(-1, deterministic.shape[-1])

        x = torch.cat([posterior, deterministic], -1)
        x = self.network(x)
        if seq:
            x = x.reshape(seq_len, batch_size, *self.observation_shape)
        dist = create_normal_dist(x, self.log_std.exp(), event_shape=len(self.observation_shape))
        return dist
