import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def create_temporal_windows(frames, labels, window_size=16, stride=8):
    """
    Create overlapping temporal windows for sequence learning.
    - frames: sequence-like (list or array) of frame arrays, length F
    - labels: sequence-like of labels, length F
    - window_size: number of consecutive frames per window
    - stride: step between window starts
    Returns: (windows, window_labels) as numpy arrays.
    """
    if len(frames) != len(labels):
        raise ValueError("frames and labels must have the same length")

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers")

    if len(frames) < window_size:
        # No full windows possible
        return np.empty((0,)), np.empty((0,))

    windows = []
    window_labels = []

    # include last window that fits: +1 to include endpoint
    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i + window_size]
        label = labels[i + window_size // 2]  # Use middle frame label

        windows.append(np.stack(window))  # ensure each window is an array (T, H, W, C) or (T, H, W, 1)
        window_labels.append(label)

    return np.array(windows), np.array(window_labels)

class MIntPAINDataset(Dataset):
    """
    PyTorch Dataset for multimodal temporal windows.
    Expects numpy arrays with shapes:
      rgb_windows: (N, T, H, W, C)
      depth_windows: (N, T, H, W, 1)
      thermal_windows: (N, T, H, W, 1)
      labels: (N,) or list-like
    """
    def __init__(self, rgb_windows, depth_windows, thermal_windows, labels):
        self.rgb = np.asarray(rgb_windows)
        self.depth = np.asarray(depth_windows)
        self.thermal = np.asarray(thermal_windows)
        self.labels = np.asarray(labels)

        if not (len(self.rgb) == len(self.depth) == len(self.thermal) == len(self.labels)):
            raise ValueError("All inputs must have the same first-dimension length (N)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Each item: rgb (T,H,W,C) -> (C,T,H,W)
        rgb = torch.from_numpy(self.rgb[idx]).float().permute(3, 0, 1, 2)
        depth = torch.from_numpy(self.depth[idx]).float().permute(3, 0, 1, 2)
        thermal = torch.from_numpy(self.thermal[idx]).float().permute(3, 0, 1, 2)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)  # scalar long

        return rgb, depth, thermal, label

def create_dataloaders(train_dataset, val_dataset, batch_size=4, shuffle_train=True, num_workers=0):
    """
    Helper to create PyTorch DataLoaders. Returns (train_loader, val_loader).
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

# Example usage (replace train_dataset/val_dataset with actual MIntPAINDataset instances):
# train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=4)
