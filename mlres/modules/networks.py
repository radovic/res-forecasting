import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mlres.modules.layers import DilatedConvLayer1d

class WaveNet(nn.Module):
    def __init__(self, n_layers, n_input_channels, n_channels, window_size, horizon, causal_convolutions=True):
        super(WaveNet, self).__init__()
        self.n_layers = n_layers
        self.n_input_channels = n_input_channels
        self.n_channels = n_channels
        self.kernel_size = 2
        self.window_size = window_size
        self.horizon = horizon
        self.causal_convolutions = causal_convolutions

        def initialize_weights(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        dilations = 2 ** np.arange(n_layers)
        self.layers = nn.ModuleList(
            [DilatedConvLayer1d(in_channels=n_input_channels, out_channels=n_channels, kernel_size=self.kernel_size, dilation=dilations[0])] +
            [DilatedConvLayer1d(in_channels=n_channels, out_channels=n_channels, kernel_size=self.kernel_size, dilation=d) for d in dilations[1:]] +
            [nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size=1)]
        )

        self.layers[-1].apply(initialize_weights)


    @property
    def receptive_field(self):
        return 2 ** (self.n_layers - 1) * self.kernel_size

    def forward(self, x):
        padding = self.receptive_field - 1 - (self.window_size - self.horizon)
        out = F.pad(x, (padding, 0)) if self.causal_convolutions else x
        for layer in self.layers:
            out = layer(out)
        return out
