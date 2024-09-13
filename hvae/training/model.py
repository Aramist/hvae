from math import ceil, floor
from typing import Optional

import torch
from numpy import sqrt
from torch import nn
from torch.nn import functional as F

sqrt2 = sqrt(2.0)


def same_padding(kernel_size: int, dilation: int, stride: int) -> int:
    """Computes the amount of padding needed on either side of a 1d signal
    to preserve it's length after convolution
    """
    # length formula: l_out = floor[  (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1  ]

    if stride == 1:
        pad_amt = int((dilation * (kernel_size - 1) + 1 - stride) // 2)
    else:
        pad_amt = int((dilation * (kernel_size - 1) + 1 - stride) // 2) + 2
    return pad_amt


class Layer(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        channels: int,
        dilation: int,
        stride: int,
    ):
        super(Layer, self).__init__()
        self.dilated_conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=same_padding(kernel_size, dilation, stride),
        )
        self.output_conv = nn.Conv1d(channels, channels * 2, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, channels, time]
        # t: [batch, time_step_dim]
        # c: [batch, conditioning_dim, time]
        dc = self.dilated_conv(x)
        dc_tanh, dc_sigmoid = dc.chunk(2, dim=1)
        dc = torch.tanh(dc_tanh) * torch.sigmoid(dc_sigmoid)
        residual, readout = torch.chunk(self.output_conv(dc), 2, dim=1)
        # An unofficial implementation scales the residual by 2**-.5, but this detail doesn't
        # seem to be mentioned in the original paper
        return (residual + x) / sqrt2, readout


class Block(nn.Module):
    def __init__(
        self,
        num_layers: int,
        kernel_size: int,
        dilation: int,
        channels_in: int,
        channels_out: int,
        downsample_factor: int,
    ):
        super(Block, self).__init__()

        if num_layers < 1:
            raise ValueError("Block: num_layers must be >= 1")

        self.layers = nn.ModuleList(
            [
                Layer(
                    kernel_size=kernel_size,
                    channels=channels_in,
                    dilation=dilation,
                    stride=1,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_conv = nn.Conv1d(
            in_channels=channels_in,
            out_channels=channels_out,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=downsample_factor,
            padding=same_padding(kernel_size, dilation, downsample_factor),
        )

        self.scale_factor = sqrt(num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        readout_sum = None
        for layer in self.layers:
            result = layer(x)
            if readout_sum is None:
                x, readout_sum = result
            else:
                x, readout = result
                readout_sum = readout_sum + readout

        readout_sum = readout_sum / self.scale_factor
        return self.final_conv(readout_sum)


class TransposeBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        kernel_size: int,
        dilation: int,
        channels_in: int,
        channels_out: int,
        downsample_factor: int,
    ):
        super(TransposeBlock, self).__init__()

        if num_layers < 1:
            raise ValueError("Block: num_layers must be >= 1")

        self.initial_conv = nn.ConvTranspose1d(
            in_channels=channels_in,
            out_channels=channels_out,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=downsample_factor,
            padding=same_padding(kernel_size, dilation, downsample_factor),
            output_padding=downsample_factor - 1,
        )

        self.layers = nn.ModuleList(
            [
                Layer(
                    kernel_size=kernel_size,
                    channels=channels_out,
                    dilation=dilation,
                    stride=1,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_conv = nn.Conv1d(
            in_channels=channels_out,
            out_channels=channels_out,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            padding=same_padding(kernel_size, dilation, downsample_factor),
        )

        self.scale_factor = sqrt(num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        readout_sum = None
        x = self.initial_conv(x)
        for layer in self.layers:
            result = layer(x)
            if readout_sum is None:
                x, readout_sum = result
            else:
                x, readout = result
                readout_sum = readout_sum + readout

        readout_sum = readout_sum / self.scale_factor
        return self.final_conv(readout_sum)


class OBO(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        use_residual: bool = True,
        use_batch_norm: bool = True,
        use_dropout: bool = False,
    ):
        super(OBO, self).__init__()
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels, kernel_size=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(in_channels) if use_batch_norm else nn.Identity(),
                    nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
                )
                for _ in range(num_layers - 1)
            ]
            + [
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = (x + residual) / sqrt2

        return x


class WaveNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers_per_block: int,
        kernel_size: list[int],
        num_channels: list[int],
        dilation: list[int],
        downsample_factors: list[int],
        latent_dim: list[int],
        *,
        lp_num_layers: int = 2,
        lp_use_residual: bool = False,
        lp_use_batch_norm: bool = True,
        lp_use_dropout: bool = False,
    ):
        super(WaveNet, self).__init__()
        self.input_dim = input_dim
        self.kernel_sizes = kernel_size
        self.num_channels = num_channels
        self.dilations = dilation
        self.downsample_factors = downsample_factors
        self.latent_dims = latent_dim

        self.encoders = nn.ModuleList(
            [
                Block(
                    num_layers=num_layers_per_block,
                    kernel_size=K,
                    channels_in=C,
                    channels_out=C,
                    dilation=D,
                    downsample_factor=S,
                )
                for K, C, D, S in zip(
                    self.kernel_sizes,
                    self.num_channels,
                    self.dilations,
                    self.downsample_factors,
                )
            ]
        )

        self.decoders = nn.ModuleList(
            [
                TransposeBlock(
                    num_layers=num_layers_per_block,
                    kernel_size=K,
                    channels_in=C,
                    channels_out=C,
                    dilation=D,
                    downsample_factor=S,
                )
                for K, C, D, S in zip(
                    self.kernel_sizes,
                    self.num_channels,
                    self.dilations,
                    self.downsample_factors,
                )
            ]
        )

        self.latent_parametrization = nn.ModuleList(
            [
                OBO(
                    lp_num_layers,
                    Ci,
                    Cl * 2,  # diagonal gaussian
                    use_residual=lp_use_residual,
                    use_batch_norm=lp_use_batch_norm,
                    use_dropout=lp_use_dropout,
                )
                for Ci, Cl in zip(self.num_channels, self.latent_dims)
            ]
        )

        self.latent_unparametrization = nn.ModuleList(
            [
                OBO(
                    lp_num_layers,
                    Cl,
                    Ci,
                    use_residual=lp_use_residual,
                    use_batch_norm=lp_use_batch_norm,
                    use_dropout=lp_use_dropout,
                )
                for Ci, Cl in zip(self.num_channels, self.latent_dims)
            ]
        )

        self.input_conv = nn.Conv1d(input_dim, self.num_channels[0], kernel_size=1)
        self.output_conv = nn.Conv1d(self.num_channels[0], input_dim, kernel_size=1)

    def _clip_gradients(self, grad_norm: float = 1.0) -> bool:
        """Attempts to clip the gradients of the model's parameters
        to a specified norm and returns True if suceeded"""
        try:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), grad_norm, error_if_nonfinite=True
            )
        except:
            return False
        return True

    def encode(
        self, x: torch.Tensor, num_timescales: int = 1
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns the parameters for the latent distribution(s)
        x: [batch, channels, time]
        num_timescales: number of timescales to use in the computation. Can be gradually increased throughout training

        Returns: [(mean, logvar)_i, ...] for each time scale, from finest to coarsest
        """
        x = F.relu(self.input_conv(x))
        latent_params = []

        for i in range(num_timescales):
            enc = self.encoders[i]  # downsamples T -> T/S
            parametrization = self.latent_parametrization[i]  # -> mean, logvar

            x = enc(x)
            param = parametrization(x)
            mean, logvar = torch.chunk(param, 2, dim=1)
            latent_params.append((mean, logvar))

        return latent_params

    def sample(
        self, latent_params: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Samples from the latent distribution(s)"""

        samples = []
        for mean, logvar in latent_params:
            # dist = torch.distributions.MultivariateNormal(mean, torch.diag(torch.exp(logvar)))
            # z = dist.rsample()
            z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            samples.append(z)

    def decode(
        self, latent_samples: list[torch.Tensor], num_timescales: int = 1
    ) -> torch.Tensor:
        """Computes the mean of the output distribution p(x | z)"""
        x = None
        for i in range(num_timescales):
            dec = self.decoders[-i]
            expander = self.latent_unparametrization[-i]

            mean, logvar = latent_samples[-i]
            z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            expz = expander(z)
            if x is None:
                x = dec(expz)
            else:
                x = x + dec(expz)
        return self.output_conv(x)


if __name__ == "__main__":
    # Test to see if all the shapes look good
    test_model = WaveNet(
        input_dim=80,
        num_layers_per_block=2,
        kernel_size=[7, 7, 7, 7],
        num_channels=[64, 64, 64, 64],
        dilation=[3, 3, 2, 2],
        downsample_factors=[16, 16, 16, 16],
        latent_dim=[16, 16, 16, 16],
    )

    # Simulate a batch of mel spectrogram
    test_data = torch.randn(4, 80, int(2**16))
    print("Input shape: [batch, channels, time]:")
    print(test_data.shape)

    # Test the encoder
    latent_params = test_model.encode(test_data, num_timescales=4)
    print("Latent params:")
    for mean, logvar in latent_params:
        print(mean.shape, logvar.shape)

    # Test the decoder
    reconstruction = test_model.decode(latent_params)
    print("Reconstruction:")
    print(reconstruction.shape)
