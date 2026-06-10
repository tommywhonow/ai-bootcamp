import torch
import torch.nn as nn

# A fake 'image': batch of 1, 1 channel, 8x8 pixels
image = torch.randn(1, 1, 8, 8)             # (batch, channels, height, width)
print(f"Input shape:    {image.shape}")     # ( 1,      1,          8,    8)

# CONVOLUTION -1 in, 4 filters, 3x3 window, padding 1
conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
after_conv = conv(image)
print(f"After conv:     {after_conv.shape}")    # (1, 4, 8, 8)
#   4 channels now (one feature map per filter); still 8x8 (padding kept size)

#  POOLING - 2x2 max pool halves height and width
pool = nn.MaxPool2d(kernel_size=2)
after_pool = pool(after_conv)
print(f"After pool:     {after_pool.shape}") # (1, 4, 8, 8)

# FLATTEN - unroll everything except the batch dimension
flat = after_pool.view(after_pool.size(0), -1)
print(f"After flatten:      {flat.shape}")      #(1, 64)
# 4 channels x 4 x 4 = 64 numbers, ready for a linear layer

# RELU works on conv output exactly like before
relu = nn.ReLU()
print(f"Relu min value:  {relu(after_conv).min().item():.1f}") # 0.0