import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class HardTaskDataset(Dataset):
    def __init__(self, audio_data, text_data, labels):
        self.audio = audio_data
        self.text = text_data
        
        # Process Labels for CVAE (One-Hot)
        self.raw_labels = np.array(labels).reshape(-1, 1)
        self.encoder = OneHotEncoder(sparse_output=False)
        self.one_hot_labels = self.encoder.fit_transform(self.raw_labels)
        self.n_classes = self.one_hot_labels.shape[1]

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return {
            'audio': torch.tensor(self.audio[idx], dtype=torch.float32),
            'text': torch.tensor(self.text[idx], dtype=torch.float32),
            'label_vec': torch.tensor(self.one_hot_labels[idx], dtype=torch.float32),
            'label_raw': self.raw_labels[idx][0]
        }

def get_hard_dataloaders(audio_data, text_data, labels, batch_size=32):
    dataset = HardTaskDataset(audio_data, text_data, labels)
    
    # 80/20 Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.n_classes