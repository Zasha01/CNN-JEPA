#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspace/I-JEPA-CNN')

import torch
import hydra
from omegaconf import DictConfig
from pretrain.train_ijepacnn import IJEPA_CNN

def test_ijepa_forward():
    """Test IJEPA_CNN forward pass with tactile data"""
    print("=== Testing IJEPA_CNN Forward Pass ===")
    
    # Create complete config for tactile data
    cfg = DictConfig({
        'backbone': {'name': 'tactnet', 'pretrained_weights': None, 'kwargs': {'global_pool': ''}},
        'predictor': {'n_layers': 3, 'kernel_size': 3, 'dw_sep_conv': True},
        'use_projection_head': False,
        'data': {'dataset_name': 'tactmat', 'data_path': 'data/tactmat.h5'},
        'mask': {
            'strategy': 'random',  # Force random masking only
            'mutli_block_kwargs': {},
            'mixed_mutli_block_ratio': 0.0
        },
        'mask_ratio': 0.6,
        'trainer': {'max_epochs': 101},
        'optimizer': {
            'lr': 0.001,
            'weight_decay': 0.04,
            'backbone_multiplier': 1.0,
            'predictor_multiplier': 1.0,
            'betas': [0.9, 0.95],
            'wd_scheduler': 'cosine',
            'wd_max': 0.04,
            'wd_min': 0.04,
            'warmup_epochs': 10,
            'warmup_lr': 1e-6
        },
        'loss': {
            'normalize_targets': True,
            'reconstruction_loss': 'l1'
        },
        'momentum': {'backbone': 0.996},
        'online_eval': {
            'val_dataset': {'name': 'tactmat', 'kwargs': {}},
            'linear_eval_every': 10
        },
        'log_train_images_every': 10,
        'log_val_images_every': 10,
        'finetuning': {'early_stop_patience': 10}
    })
    
    try:
        # Create IJEPA model
        print("Creating IJEPA_CNN model...")
        model = IJEPA_CNN(cfg)
        print("✅ Model created successfully")
        
        # Manually set the dimensions (normally done in setup)
        model.input_size = 16
        model.fmap_h, model.fmap_w = 1, 16
        model.len_keep = round(16 * (1 - 0.6))  # Keep 40% of patches
        print(f"Feature map: {model.fmap_h}x{model.fmap_w}, keeping {model.len_keep}/16 patches")
        
        # Test forward pass
        batch_size = 2
        sample_input = torch.randn(batch_size, 1, 1000, 16)
        print(f"Input shape: {sample_input.shape}")
        
        print("Testing forward pass...")
        with torch.no_grad():
            predictions, context_mask, target_mask = model.forward(sample_input)
            
        print(f"✅ Forward pass successful!")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Context mask shape: {context_mask.shape}")
        print(f"Target mask shape: {target_mask.shape}")
        
        # Test momentum forward
        print("Testing momentum forward pass...")
        with torch.no_grad():
            target_features = model.forward_momentum(sample_input)
        
        print(f"✅ Momentum forward pass successful!")
        print(f"Target features shape: {target_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in IJEPA forward: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing IJEPA_CNN Forward Pass...")
    success = test_ijepa_forward()
    
    if success:
        print("\n✅ IJEPA forward pass test passed!")
    else:
        print("\n❌ IJEPA forward pass test failed.") 