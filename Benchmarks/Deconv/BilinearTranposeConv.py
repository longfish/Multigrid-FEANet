import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, " + \
            "or one group per channel"

        kernel_size = (2 * stride[0] - 1, 2 * stride[1] - 1)
        padding = (stride[0] - 1, stride[1] - 1)
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant(self.bias, 0)
        nn.init.constant(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(stride)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel channel by channel
        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            # e.g. with stride = 4
            # delta = [-3, -2, -1, 0, 1, 2, 3]
            # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            delta = torch.arange(1 - channel_stride, channel_stride)
            channel_filter = (1 - torch.abs(delta / channel_stride))
            # Apply the channel filter to the current channel
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel