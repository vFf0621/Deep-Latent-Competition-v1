import torch
import torch.nn as nn

from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
)

'''

The encoder is used to encode the observations into a latent space 
then fed into the representation model. It is a CNN model.

Our contribution was the addition of batch normalization layers to 
ensure stability in training.

'''

class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.encoder

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Conv2d(
                self.observation_shape[0],
                self.config.depth * 1,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.BatchNorm2d(self.config.depth * 1),
            nn.Conv2d(
                self.config.depth * 1,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.BatchNorm2d(self.config.depth * 2),

            nn.Conv2d(
                self.config.depth * 2,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.BatchNorm2d(self.config.depth * 4),

            nn.Conv2d(
                self.config.depth * 4,
                self.config.depth * 8,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.BatchNorm2d(self.config.depth * 8),

        )
        self.bn = nn.LayerNorm(512)
        self.fc = nn.Linear(4096, 512)
        self.network.apply(initialize_weights)

    def forward(self, x, seq=0):
        if seq:
            seq_len = x.shape[0]
            batch_size = x.shape[1]
            x = x.reshape(-1, *self.observation_shape)
        else:
            x = x.unsqueeze(0)
        y = self.network(x).view(-1)
        y = self.fc(y.reshape(-1, 4096)).squeeze(0)
        y = nn.LeakyReLU()(y)
        y = self.bn(y)
        if seq:
            y = y.reshape(seq_len, batch_size, -1)
        return y