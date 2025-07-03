#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspace/I-JEPA-CNN')

import torch
import torch.nn as nn
from models.tactnet_jepa import TactNet
from data.tactmat_dataset import TactMatDataset
from pretrain.tactile_transforms import TactileIJEPATransform

def test_tactnet_forward():
    print("=== Testing TactNet Forward Pass ===")
    
    # Create TactNet
    tactnet = TactNet()
    print(f"TactNet created successfully")
    print(f"num_features: {tactnet.num_features}")
    print(f"downsample_ratio: {tactnet.get_downsample_ratio()}")
    
    # Test with sample data
    batch_size = 2
    sample_input = torch.randn(batch_size, 1, 1000, 16)
    print(f"Input shape: {sample_input.shape}")
    
    try:
        output = tactnet(sample_input)
        print(f"TactNet output shape: {output.shape}")
        
        # Test hierarchical output
        hierarchical_output = tactnet(sample_input, hierarchical=True)
        print(f"Hierarchical output shapes: {[x.shape for x in hierarchical_output]}")
        
    except Exception as e:
        print(f"ERROR in TactNet forward: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_dataset_loading():
    print("\n=== Testing Dataset Loading ===")
    
    try:
        dataset = TactMatDataset("data/tactmat.h5", split="val")
        print(f"Dataset created: {len(dataset)} samples")
        
        # Test loading one sample
        sample, label = dataset[0]
        print(f"Sample shape: {sample.shape}, Label: {label}")
        
        # Test transform
        transform = TactileIJEPATransform(normalize=True)
        transformed = transform(sample)
        print(f"Transformed shape: {transformed.shape}")
        
    except Exception as e:
        print(f"ERROR in dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_simple_masking():
    print("\n=== Testing Simple Masking ===")
    
    try:
        batch_size = 2
        fmap_h, fmap_w = 1, 16  # Tactile feature map dimensions
        
        # Create simple random mask
        total_patches = fmap_h * fmap_w
        keep_ratio = 0.4
        len_keep = round(total_patches * keep_ratio)
        
        print(f"Feature map: {fmap_h}x{fmap_w}, keeping {len_keep}/{total_patches} patches")
        
        # Random masking
        idx = torch.rand(batch_size, total_patches).argsort(dim=1)
        idx = idx[:, :len_keep]
        
        context_mask = torch.zeros(batch_size, total_patches, dtype=torch.bool)
        context_mask.scatter_(dim=1, index=idx, value=True)
        context_mask = context_mask.view(batch_size, 1, fmap_h, fmap_w)
        target_mask = context_mask.logical_not()
        
        print(f"Context mask shape: {context_mask.shape}")
        print(f"Target mask shape: {target_mask.shape}")
        print(f"Context patches kept: {context_mask.sum().item()}")
        print(f"Target patches: {target_mask.sum().item()}")
        
    except Exception as e:
        print(f"ERROR in masking: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Starting Tactile I-JEPA Debug Tests...")
    
    success = True
    success &= test_tactnet_forward()
    success &= test_dataset_loading() 
    success &= test_simple_masking()
    
    if success:
        print("\n✅ All tests passed! Individual components work correctly.")
    else:
        print("\n❌ Some tests failed. Check errors above.") 