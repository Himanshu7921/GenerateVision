# Conv learns features
# ReLU activates them
# BatchNorm stabilizes them
# MaxPool localizes them
# GlobalAvgPool summarizes them
# Linear classifies them

import torch
import torch.nn as nn

class Conv2d(nn.Module):
    """
    This class is responsible for performing a Constrained Linear Transformation on input signal
        - what's constrained on Linear Transformation (y = Wx + b)?
            - Spatial Locality is preserved
            - Weight sharing is enabled

    Note: This inplementation is differ from origianl PyTorch's implementation in some ways
    which Ways?
        - No Padding
        - No bias boolean for optional bias
        - No Dialation
    """
    def __init__(self, in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Learnable Parameter
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.tensor):
        B, C, H, W = x.shape
        assert C == self.in_channels, "self.in_channels must be equal to x.shape[-3] (C), error from Conv2d block"
        H_out = ((H - self.kernel_size) // self.stride) + 1
        W_out = ((W - self.kernel_size) // self.stride) + 1
        output = torch.zeros(B, self.out_channels, H_out, W_out, device = x.device, dtype = x.dtype)
        for b in range(B):
            for o in range(self.out_channels):
                for i in range(0, H - self.kernel_size + 1, self.stride):
                    for j in range(0, W - self.kernel_size + 1, self.stride):
                        patch = x[b][:, i: i + self.kernel_size, j: j + self.kernel_size]
                        y = torch.sum(self.weight[o] * patch) + self.bias[o]
                        output[b][o][i//self.stride, j//self.stride] = y
        return output

class MaxPool2d(nn.Module):
    """
    This class is responsible for performing a Maximum pooling independently over each channel of input signal
    """
    def __init__(self,
                kernel_size: int,
                stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.tensor):
        B, C, H, W = x.shape
        H_out = ((H - self.kernel_size) // self.stride) + 1
        W_out = ((W - self.kernel_size) // self.stride) + 1
        output = torch.zeros(B, C, H_out, W_out, device = x.device, dtype = x.dtype)
        for b in range(B):
            for c in range(C):
                for i in range(0, H - self.kernel_size + 1, self.stride):
                    for j in range(0, W - self.kernel_size + 1, self.stride):
                        region = x[b][c, i: i + self.kernel_size, j: j + self.kernel_size]
                        output[b][c][i//self.stride, j//self.stride] = region.max()
        return output

class BatchNorm2d(nn.Module):
    """
    > BatchNorm2d normalizes per channel
        - statistics are over (b, h, w)
        - parameters are per c
    > BatchNorm2d treats each channel,
      as a distribution of values over (batch x height x width) and normalizes that distribution independently
    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(in_channels))
        self.beta = torch.nn.Parameter(torch.zeros(in_channels))
        self.eps = eps
        
    def forward(self, x: torch.tensor):
        # B, C, H, W = x.shape
        mean = x.mean(dim = (0, 2, 3), keepdim = True)
        var = x.var(dim = (0, 2, 3), keepdim = True, unbiased = False)
        x_hat = (x - mean) / ((var + self.eps) ** 0.5)
        return self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)

class LinearLayer(nn.Module):
    """
    A Linear Transformation from in_channels to n_features dimension
    """
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_channels, n_classes))
        self.bias = nn.Parameter(torch.rand(n_classes))
    
    def forward(self, x):
        return x @ self.weight + self.bias
        
class GlobalAvgPool(nn.Module):
    """
    Global Average Pooling maps (B, C, H, W) to (B, C) by averaging,
    each channel's entire spatial activation map
    into a single scalar, without sliding kernels or strides.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        # we want to transform (B, C, H, W) to (B, C)
        output = torch.zeros(B, C, dtype = x.dtype).to(device = x.device)
        for b in range(B):
            for c in range(C):
                channel = x[b, c, :, :]
                y = channel.mean()
                output[b, c] = y
        return output

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(10, 3, 32, 32).to(device=device)

    print("\n" + "=" * 80)
    print("Experiment Setup")
    print("=" * 80)
    print(f"Compute Device        : {device}")
    print(f"Input Tensor Shape    : {tuple(x.shape)}")
    print("=" * 80 + "\n")

    in_channels = x.shape[-3]
    out_channels = 4

    # --------------------------------------------- Conv2d Layer ---------------------------------------------
    print(">>> Conv2d Layer")
    print("-" * 80)

    my_conv = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=1
    ).to(device)

    # self.weight.shape --> (4, 3, 2, 2)
    # self.bias.shape --> (4)
    # h_out = ((32 - 2) // 1) + 1 = 31, w_out = ((32 - 2) // 1) + 1 = 31
    # self.output.shape --> (B, output_channels, h_out, w_out) = (10, 4, 31, 31)

    output = my_conv(x)

    print(f"Output Tensor Shape   : {tuple(output.shape)}")
    print(f"Expected Output Shape : {(10, 4, 31, 31)}")
    print("-" * 80 + "\n")

    # --------------------------------------------- MaxPool Layer ---------------------------------------------
    print(">>> MaxPool2d Layer")
    print("-" * 80)

    maxpool_layer = MaxPool2d(kernel_size=2, stride=2)
    max_pool_output = maxpool_layer(output)

    # H_out = ((31 - 2) // 2 ) + 1 = 15
    # W_out = ((31 - 2) // 2 ) + 1 = 15

    print(f"Output Tensor Shape   : {tuple(max_pool_output.shape)}")
    print(f"Expected Output Shape : {(10, 4, 15, 15)}")
    print("-" * 80 + "\n")

    # --------------------------------------------- BatchNorm2d Layer ---------------------------------------------
    print(">>> BatchNorm2d Layer")
    print("-" * 80)

    batch_norm_layer = BatchNorm2d(in_channels=out_channels)
    batch_norm_output = batch_norm_layer(max_pool_output)

    # BatchNorm2d just normalizes per channels, dsn't effect the shape of input signal

    print(f"Output Tensor Shape   : {tuple(batch_norm_output.shape)}")
    print(f"Expected Output Shape : {(10, 4, 15, 15)}")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("Forward Pass Verification Completed Successfully")
    print("=" * 80 + "\n")

    # --------------------------------------------- GlobalAvgPool Layer ---------------------------------------------
    print(">>> GlobalAvgPool Layer")
    print("-" * 80)

    gap = GlobalAvgPool()
    gap_output = gap(batch_norm_output)

    # GlobalAvgPool reduces spatial dimensions (H, W) by averaging per channel
    # (B, C, H, W) → (B, C)

    print(f"Input Tensor Shape    : {tuple(batch_norm_output.shape)}")
    print(f"Output Tensor Shape   : {tuple(gap_output.shape)}")
    print(f"Expected Output Shape : {(10, 4)}")
    print("-" * 80)

