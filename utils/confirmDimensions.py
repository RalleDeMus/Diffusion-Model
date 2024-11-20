import torch
import torch.nn as nn

class DimensionTracker:
    def __init__(self, input_dim, input_channels):
        self.current_dim = input_dim  # Input dimension (H, W)
        self.current_channels = input_channels

    def conv2d(self, out_channels, kernel_size, stride, padding):
        h, w = self.current_dim
        h = (h + 2 * padding - kernel_size) // stride + 1
        w = (w + 2 * padding - kernel_size) // stride + 1
        self.current_dim = (h, w)
        self.current_channels = out_channels
        return self.current_dim, self.current_channels

    def maxpool2d(self, kernel_size, stride, padding):
        h, w = self.current_dim
        h = (h + 2 * padding - kernel_size) // stride + 1
        w = (w + 2 * padding - kernel_size) // stride + 1
        self.current_dim = (h, w)
        return self.current_dim

    def convtranspose2d(self, out_channels, kernel_size, stride, padding, output_padding):
        h, w = self.current_dim
        h = (h - 1) * stride - 2 * padding + kernel_size + output_padding
        w = (w - 1) * stride - 2 * padding + kernel_size + output_padding
        self.current_dim = (h, w)
        self.current_channels = out_channels
        return self.current_dim, self.current_channels


def calculate_dimensions():
    input_dim = (32, 32)  # Input dimensions (height, width)
    input_channels = 1
    batch_size = 1

    # Initialize parameters from ScoreNetwork
    chs = [32, 64, 128, 256, 256]
    num_groups = 8
    t_dim = 1  # Time dimension concatenated to the input

    current_dim = input_dim
    current_channels = input_channels + t_dim

    print(f"Initial input dimensions: {current_dim} with {current_channels} channels")

    # Encoder layers
    encoder_layers = [
        {"type": "conv2d", "out_channels": chs[0], "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2, "padding": 0},
        {"type": "conv2d", "out_channels": chs[1], "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2, "padding": 0},
        {"type": "conv2d", "out_channels": chs[2], "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2, "padding": 0},
        {"type": "conv2d", "out_channels": chs[3], "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2, "padding": 0},
        {"type": "conv2d", "out_channels": chs[4], "kernel_size": 3, "stride": 1, "padding": 1},
    ]

    # Decoder layers
    decoder_layers = [
        {"type": "convtranspose2d", "out_channels": chs[3], "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        {"type": "convtranspose2d", "out_channels": chs[2], "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        {"type": "convtranspose2d", "out_channels": chs[1], "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        {"type": "convtranspose2d", "out_channels": chs[0], "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        {"type": "conv2d", "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1},
    ]

    # Process encoder
    print("\nEncoder:")
    for i, layer in enumerate(encoder_layers):
        if layer["type"] == "conv2d":
            current_dim = (
                (current_dim[0] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1,
                (current_dim[1] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1,
            )
            current_channels = layer["out_channels"]
        elif layer["type"] == "maxpool2d":
            current_dim = (
                (current_dim[0] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1,
                (current_dim[1] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1,
            )
        print(f"Layer {i}: Output Dimension = {current_dim}, Channels = {current_channels}")

    # Process decoder
    print("\nDecoder:")
    for i, layer in enumerate(decoder_layers):
        if layer["type"] == "convtranspose2d":
            current_dim = (
                (current_dim[0] - 1) * layer["stride"] - 2 * layer["padding"] + layer["kernel_size"] + layer["output_padding"],
                (current_dim[1] - 1) * layer["stride"] - 2 * layer["padding"] + layer["kernel_size"] + layer["output_padding"],
            )
            current_channels = layer["out_channels"]
        elif layer["type"] == "conv2d":
            current_dim = (
                (current_dim[0] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1,
                (current_dim[1] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1,
            )
            current_channels = layer["out_channels"]
        print(f"Layer {i}: Output Dimension = {current_dim}, Channels = {current_channels}")

calculate_dimensions()
