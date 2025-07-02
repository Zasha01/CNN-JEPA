# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in under
# https://github.com/keyu-tian/SparK/blob/main/LICENSE ur.

import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model


class YourConvNet(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    
    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        raise NotImplementedError
    
    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        raise NotImplementedError
    
    def forward(self, inp_bchw: torch.Tensor, hierarchical=False):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        raise NotImplementedError


class TactNet(nn.Module):
    def __init__(self, num_classes=36, **kwargs):
        super().__init__()
        # Feature extractor for tactile data: (B, 1, 1000, 16) -> (B, 128, 1, 16)
        # Based on original TactNet implementation that expects flattened spatial data
        self.features = nn.Sequential(
            # First conv: (B, 1, 1000, 16) -> (B, 32, 1000, 16)
            nn.Conv2d(1, 32, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(10, 1), stride=(10, 1)),  # -> (B, 32, 100, 16)

            # Second conv: (B, 32, 100, 16) -> (B, 64, 100, 16)
            nn.Conv2d(32, 64, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(10, 1), stride=(10, 1)),  # -> (B, 64, 10, 16)

            # Third conv: (B, 64, 10, 16) -> (B, 128, 10, 16)
            nn.Conv2d(64, 128, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(10, 1), stride=(10, 1)),  # -> (B, 128, 1, 16)
        )
        self.num_features = 128  # Output channels of last conv layer

    def get_downsample_ratio(self) -> int:
        # For JEPA masking, we want to think of the 16 spatial features as a 4x4 grid
        # So the effective downsample ratio should be 4 (16 -> 4x4 conceptually)
        # This allows JEPA to mask patches in the spatial domain
        return 4  # Treat as 4x4 spatial grid for masking purposes

    def get_feature_map_channels(self) -> List[int]:
        # Output channels after each MaxPool2d
        return [32, 64, 128]

    def forward(self, x, hierarchical=False):
        # Input x: (B, 1, 1000, 16) - tactile data with flattened spatial dimensions
        
        # Ensure input has correct shape
        if len(x.shape) == 5 and x.shape[-1] == 4 and x.shape[-2] == 4:
            # If input is (B, 1, 1000, 4, 4), flatten spatial dimensions to (B, 1, 1000, 16)
            B, C, T, H, W = x.shape
            x = x.view(B, C, T, H * W)  # -> (B, 1, 1000, 16)
        elif len(x.shape) == 3:
            # If input is (B, 1000, 16), add channel dimension
            x = x.unsqueeze(1)  # -> (B, 1, 1000, 16)
        
        feats = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
        
        if hierarchical:
            return feats  # List of feature maps
        else:
            return x  # Final feature map: (B, 128, 1, 16)


@register_model
def tactnet(pretrained=False, **kwargs):
    return TactNet(**kwargs)


@register_model
def your_convnet_small(pretrained=False, **kwargs):
    raise NotImplementedError
    return YourConvNet(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('tactnet')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # Test with tactile data dimensions (flattened spatial)
    B, C, T, spatial_features = 4, 1, 1000, 16
    inp = torch.rand(B, C, T, spatial_features)
    
    # Test hierarchical output
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print("Hierarchical feature shapes:", [tuple(t.shape) for t in feats])
    
    # Test final output shape
    final_output = cnn(inp, hierarchical=False)
    print(f"Final output shape: {final_output.shape}")
    
    # Test with 4x4 spatial input (should be automatically flattened)
    inp_4x4 = torch.rand(B, C, T, 4, 4)
    final_output_4x4 = cnn(inp_4x4, hierarchical=False)
    print(f"Final output shape from 4x4 input: {final_output_4x4.shape}")
    
    # Check the channel numbers
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4  # (B, C, H, W)
        assert feat.shape[1] == ch
        print(f"Feature shape: {feat.shape}, Expected channels: {ch}")


if __name__ == '__main__':
    convnet_test()
