# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List


class TactMatDataset(Dataset):
    """Dataset class for loading tactile material data from HDF5 file.
    
    The tactmat.h5 file contains:
    - "samples": [materials=36][samples=100][time_steps=1000][taxels_x=4][taxels_y=4]
    - "materials": list of 36 material names
    """
    
    def __init__(
        self, 
        root: str, 
        transform: Optional[Callable] = None,
        split: str = "train",
        train_ratio: float = 0.8
    ):
        """
        Args:
            root: Path to the tactmat.h5 file
            transform: Optional transform to be applied on a sample
            split: "train" or "val" 
            train_ratio: Ratio of data to use for training
        """
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        # Load data from HDF5 file
        self.hdf5_file = h5py.File(root, 'r')
        self.samples = self.hdf5_file['samples']  # [36, 100, 1000, 4, 4]
        self.materials = self.hdf5_file['materials']
        
        # Get data dimensions
        self.num_materials, self.samples_per_material, self.time_steps, self.taxels_x, self.taxels_y = self.samples.shape
        
        # Create train/val split
        samples_per_split = int(self.samples_per_material * self.train_ratio)
        
        if split == "train":
            self.sample_indices = list(range(samples_per_split))
        else:  # val
            self.sample_indices = list(range(samples_per_split, self.samples_per_material))
        
        # Create flat index for all samples
        self.flat_indices = []
        for material_idx in range(self.num_materials):
            for sample_idx in self.sample_indices:
                self.flat_indices.append((material_idx, sample_idx))
    
    def __len__(self) -> int:
        return len(self.flat_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        material_idx, sample_idx = self.flat_indices[idx]
        
        # Load tactile data: [time_steps=1000, taxels_x=4, taxels_y=4]
        tactile_data = self.samples[material_idx, sample_idx]  # [1000, 4, 4]
        
        # Flatten spatial dimensions to match original TactNet: [1000, 16]
        tactile_flattened = tactile_data.reshape(1000, 16)
        
        # Convert to tensor and add channel dimension: [1, 1000, 16]
        tactile_tensor = torch.from_numpy(tactile_flattened).float().unsqueeze(0)
        
        # Apply transform if provided
        if self.transform:
            tactile_tensor = self.transform(tactile_tensor)
        
        return tactile_tensor, material_idx
    
    def get_material_names(self) -> List[str]:
        """Return list of material names."""
        return [name.decode('utf-8') if isinstance(name, bytes) else name for name in self.materials]
    
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


if __name__ == "__main__":
    # Test the dataset
    import os
    # Check if we're in the data directory or need to look in data/
    if os.path.exists("tactmat.h5"):
        dataset_path = "tactmat.h5"
    elif os.path.exists("data/tactmat.h5"):
        dataset_path = "data/tactmat.h5"
    else:
        raise FileNotFoundError("tactmat.h5 not found in current directory or data/ directory")
    
    # Create train and val datasets
    train_dataset = TactMatDataset(dataset_path, split="train")
    val_dataset = TactMatDataset(dataset_path, split="val")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Total materials: {train_dataset.num_materials}")
    print(f"Material names: {train_dataset.get_material_names()}")
    
    # Test loading a sample
    sample, label = train_dataset[0]
    print(f"Sample shape: {sample.shape}")  # Should be [1, 1000, 16]
    print(f"Label: {label}")
    
    train_dataset.close()
    val_dataset.close() 