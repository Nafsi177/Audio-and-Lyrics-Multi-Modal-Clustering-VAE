import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, audio_data, text_data):
        self.audio = torch.as_tensor(audio_data, dtype=torch.float32)
        self.text = torch.as_tensor(text_data, dtype=torch.float32)

    def __len__(self):
        return len(self.audio)
    
    def __getitem__(self, idx):
        return {
            "audio": self.audio[idx],
            "text": self.text[idx]
        }

def get_dataloaders(audio_data, text_data, batch_size=32, val_split=0.1):
    num_samples = len(audio_data)
    idx = np.random.permutation(num_samples)
    split = int((1 - val_split) * num_samples)
    
    train_idx, val_idx = idx[:split], idx[split:]
    
    train_ds = MultimodalDataset(audio_data[train_idx], text_data[train_idx])
    val_ds = MultimodalDataset(audio_data[val_idx], text_data[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader