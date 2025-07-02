#!/usr/bin/env python3

import os
from torchvision import datasets

def binary_loader(img_path):
    with open(img_path, 'rb') as img_f:
        binary_data = img_f.read()
        return binary_data

def test_imagefolder_with_symlinks():
    val_path = '/home/zakariea_sharfeddine_gmail_com/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8/imagenet-100/val'
    train_path = '/home/zakariea_sharfeddine_gmail_com/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8/imagenet-100/train'
    
    print(f"Testing val path: {val_path}")
    print(f"Directory exists: {os.path.exists(val_path)}")
    print(f"Is directory: {os.path.isdir(val_path)}")
    
    # List directory contents
    print(f"Contents: {os.listdir(val_path)[:5]}")  # First 5 items
    
    # Check first subdirectory
    first_subdir = os.path.join(val_path, 'n01440764')
    print(f"First subdir exists: {os.path.exists(first_subdir)}")
    print(f"First subdir is dir: {os.path.isdir(first_subdir)}")
    print(f"First subdir is symlink: {os.path.islink(first_subdir)}")
    
    try:
        print("Attempting to create ImageFolder for val...")
        val_dataset = datasets.ImageFolder(root=val_path, loader=binary_loader)
        print(f"Success! Val dataset length: {len(val_dataset)}")
    except Exception as e:
        print(f"Error with val dataset: {e}")
    
    try:
        print("Attempting to create ImageFolder for train...")
        train_dataset = datasets.ImageFolder(root=train_path, loader=binary_loader)
        print(f"Success! Train dataset length: {len(train_dataset)}")
    except Exception as e:
        print(f"Error with train dataset: {e}")

if __name__ == "__main__":
    test_imagefolder_with_symlinks() 