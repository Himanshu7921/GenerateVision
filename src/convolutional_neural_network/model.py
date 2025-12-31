import torch
import torch.nn as nn

from layers import Conv2d, BatchNorm2d, MaxPool2d, LinearLayer, GlobalAvgPool

class CNNModel(nn.Module):
    """
    Convolutional Neural Network implemented from first principles.
    All layers (Conv2d, BatchNorm2d, MaxPool2d) are manually derived
    and implemented without using high-level PyTorch abstractions.

    > Input (3, 32, 32)

    # Block-1: (3, 32, 32) → (16, 14, 14)
    → Conv(3 → 16) → BN → ReLU
    → Conv(16 → 16) → BN → ReLU
    → MaxPool(2 x 2)          → (16, 14, 14)

    # Block-2: (16, 14, 14) → (32, 5, 5)
    → Conv(16 → 32) → BN → ReLU
    → Conv(32 → 32) → BN → ReLU
    → MaxPool(2 x 2)          → (32, 5, 5)

    # Head: (32, 5, 5) → (32, 10)
    → GlobalAveragePool     → (32,)
    → Linear(32 → 10)
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

        # Block-1: (3, 32, 32) → (16, 14, 14)
        self.conv1 = Conv2d(
            in_channels = 3,
            out_channels = 16,
            kernel_size = 3,
            stride = 1
        )
        self.bn_1 = BatchNorm2d(in_channels = 16)
        self.conv2 = Conv2d(
            in_channels = 16,
            out_channels = 16,
            kernel_size = 3,
            stride = 1
        )
        self.bn_2 = BatchNorm2d(in_channels = 16)
        self.maxpool_1 = MaxPool2d(kernel_size = 2, stride = 2)
    
        # Block-2: (16, 14, 14) → (32, 5, 5)
        self.conv3 = Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            stride = 1
        )
        self.bn_3 = BatchNorm2d(in_channels = 32)
        self.conv4 = Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            stride = 1
        )
        self.bn_4 = BatchNorm2d(in_channels = 32)
        self.maxpool_2 = MaxPool2d(kernel_size = 2, stride = 2)

        # Head: (32, 5, 5) → (32, 10)
        self.global_avg_pool = GlobalAvgPool()
        self.linear_layer = LinearLayer(in_channels = 32,  n_classes = self.n_classes)

    def forward(self, x):
        # Block-1
        # print("Block-1")
        # print(f"Input Shape: {tuple(x.shape)}")
        con1_out = self.conv1(x)
        # print(f"con1_out Shape: {tuple(con1_out.shape)}")
        bn_1_out = self.bn_1(con1_out)
        # print(f"bn_1_out Shape: {tuple(bn_1_out.shape)}")
        relu_1_out = torch.relu(bn_1_out)
        # print(f"relu_1_out Shape: {tuple(relu_1_out.shape)}")
        con2_out = self.conv2(relu_1_out)
        # print(f"con2_out Shape: {tuple(con2_out.shape)}")
        bn_2_out = self.bn_2(con2_out)
        # print(f"bn_2_out Shape: {tuple(bn_2_out.shape)}")
        relu_2_out = torch.relu(bn_2_out)
        # print(f"relu_2_out Shape: {tuple(relu_2_out.shape)}")
        maxpool_1_out = self.maxpool_1(relu_2_out)
        # print(f"maxpool_1_out Shape: {tuple(maxpool_1_out.shape)}")
        # print(f"Block-1's Final output shape: {tuple(maxpool_1_out.shape)}\n\n")
        # print("-" * 100)

        # Block-2
        # print("\n\nBlock-2")
        con3_out = self.conv3(maxpool_1_out)
        # print(f"con3_out.shape = {tuple(con3_out.shape)}")
        bn_3_out = self.bn_3(con3_out)
        # print(f"bn_3_out.shape = {tuple(bn_3_out.shape)}")
        relu_3_out = torch.relu(bn_3_out)
        # print(f"relu_3_out.shape = {tuple(relu_3_out.shape)}")
        con4_out = self.conv4(relu_3_out)
        # print(f"con4_out.shape = {tuple(con4_out.shape)}")
        bn_4_out = self.bn_4(con4_out)
        # print(f"bn_4_out.shape = {tuple(bn_4_out.shape)}")
        relu_4_out = torch.relu(bn_4_out)
        # print(f"relu_4_out.shape = {tuple(relu_4_out.shape)}")
        maxpool_2_out = self.maxpool_2(relu_4_out)
        # print(f"maxpool_2_out.shape = {tuple(maxpool_2_out.shape)}")
        # print(f"Block-2's Final output shape: {tuple(maxpool_2_out.shape)}\n\n")
        # print("-" * 100)

        # Head
        global_avg_pool_out = self.global_avg_pool(maxpool_2_out)
        # print(f"global_avg_pool_out.shape = {tuple(global_avg_pool_out.shape)}")
        linear_layer_output = self.linear_layer(global_avg_pool_out)
        # print(f"linear_layer_output.shape = {tuple(linear_layer_output.shape)}")
        return linear_layer_output

if __name__ == "__main__":
    model = CNNModel(n_classes = 10)
    x = torch.rand(1, 3, 32, 32)
    logits = model(x)
    print(logits)
