import torch
import math
import torch.nn as nn

class DilatedConvLayer1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConvLayer1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Define the method to initialize the weights using the distribution by He et al.
        def initialize_weights(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        # Define the convolutional layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.activation = nn.ReLU() # nn.LeakyReLU(0.1) # nn.ReLU()
        # Define the residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1)
        # Initialize the weights
        self.conv.apply(initialize_weights)
        self.residual.apply(initialize_weights)


    def forward(self, x):
        x1 = self.activation(self.conv(x))
        x2 = self.residual(x[:, :, self.dilation:])
        return x1 + x2
