import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GTZAN_Dataset(Dataset):
    def __init__(self, arr):
        self.arr = torch.as_tensor(arr, dtype=torch.float32)

    def __len__(self):
        return self.arr.shape[0]
    
    def __getitem__(self, idx):
        return self.arr[idx]
    
def get_dataloaders(features,batch_size=64,val_split=0.1,shuffle=True,):
    number_of_samples = features.shape[0]
    idx = np.random.permutation(number_of_samples)

    split = int((1 - val_split) * number_of_samples)
    train_idx, val_idx = idx[:split], idx[split:]

    train_loader = DataLoader(
        GTZAN_Dataset(features[train_idx]),
        batch_size=batch_size,
        shuffle=shuffle,
    )

    val_loader = DataLoader(
        GTZAN_Dataset(features[val_idx]),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader

if __name__=="__main__":
    from preprocess import get_processed_features
    from feature_engineering import get_features
    features,genres,paths=get_features()
    scaled_features=get_processed_features(features)
    train_loader,test_loader=get_dataloaders(scaled_features)


