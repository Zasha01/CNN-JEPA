# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class TactileTransform(nn.Module):
    """Transform for tactile data that applies appropriate augmentations.
    
    Input: (C, T, F) where C=1, T=1000 (time steps), F=16 (flattened spatial features)
    Output: Same shape with applied augmentations
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        temporal_shift_max: int = 50,
        spatial_flip_prob: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.temporal_shift_max = temporal_shift_max
        self.spatial_flip_prob = spatial_flip_prob
        self.normalize = normalize
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations and return two views of the same data.
        
        Args:
            x: Input tensor of shape (C, T, F) where F=16 (flattened spatial)
            
        Returns:
            Tuple of two augmented views
        """
        # Create two different augmented views
        view1 = self._apply_augmentations(x)
        view2 = self._apply_augmentations(x)
        
        return view1, view2
    
    def _apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to tactile data."""
        x = x.clone()
        
        # 1. Add small amount of noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # 2. Random temporal shift (circular)
        if self.temporal_shift_max > 0:
            shift = np.random.randint(-self.temporal_shift_max, self.temporal_shift_max + 1)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=1)  # Shift along time dimension
        
        # 3. Random spatial permutations (since spatial data is flattened)
        if np.random.random() < self.spatial_flip_prob:
            # Reshape to 4x4, apply spatial flips, then flatten back
            C, T, F = x.shape
            x_spatial = x.view(C, T, 4, 4)  # Reshape to spatial
            
            # Horizontal flip
            if np.random.random() < 0.5:
                x_spatial = torch.flip(x_spatial, dims=[3])  # Flip along width
            
            # Vertical flip
            if np.random.random() < 0.5:
                x_spatial = torch.flip(x_spatial, dims=[2])  # Flip along height
            
            x = x_spatial.view(C, T, F)  # Flatten back
        
        # 4. Normalize if requested
        if self.normalize:
            # Normalize per sample
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True) + 1e-8
            x = (x - mean) / std
        
        return x


class TactileIJEPATransform(nn.Module):
    """IJEPA-style transform for tactile data that returns the input without modification
    for use with masked modeling approaches.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized tactile data for IJEPA training.
        
        Args:
            x: Input tensor of shape (C, T, F) where F=16 (flattened spatial)
            
        Returns:
            Normalized input tensor
        """
        if self.normalize:
            # Normalize per sample
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True) + 1e-8
            x = (x - mean) / std
        
        return x


if __name__ == "__main__":
    # Test the transforms
    batch_size = 4
    channels = 1
    time_steps = 1000
    spatial_features = 16  # Flattened 4x4
    
    # Create sample tactile data
    x = torch.randn(batch_size, channels, time_steps, spatial_features)
    
    # Test TactileTransform
    transform = TactileTransform()
    view1, view2 = transform(x)
    print(f"Input shape: {x.shape}")
    print(f"View1 shape: {view1.shape}")
    print(f"View2 shape: {view2.shape}")
    
    # Test TactileIJEPATransform
    ijepa_transform = TactileIJEPATransform()
    normalized_x = ijepa_transform(x)
    print(f"IJEPA transform output shape: {normalized_x.shape}")
    
    print("Transforms working correctly!") 