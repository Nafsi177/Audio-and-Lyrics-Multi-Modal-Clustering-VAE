import torch
import numpy as np
from tqdm import tqdm
from src.medium_task.loss import HybridLoss
import os
import matplotlib.pyplot as plt

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    metrics = {'loss': 0, 'rec_a': 0, 'rec_t': 0, 'kld': 0}
    
    for batch in loader:
        batch['audio'] = batch['audio'].to(device)
        if batch['text'] is not None:
            batch['text'] = batch['text'].to(device)
            
        optimizer.zero_grad()
        
        if batch['text'] is not None:
            x_recon_a, x_recon_t, mu, logvar = model(batch['audio'], batch['text'])
        else:
            x_recon_a, x_recon_t, mu, logvar = model(batch['audio'])
            
        loss, rec_a, rec_t, kld = loss_fn(batch, x_recon_a, x_recon_t, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        metrics['loss'] += loss.item()
        metrics['rec_a'] += rec_a.item()
        metrics['rec_t'] += rec_t.item()
        metrics['kld'] += kld.item()
        
    # Average over number of batches
    for key in metrics:
        metrics[key] /= len(loader)
    return metrics

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    metrics = {'loss': 0, 'rec_a': 0, 'rec_t': 0, 'kld': 0}
    
    with torch.no_grad():
        for batch in loader:
            batch['audio'] = batch['audio'].to(device)
            if batch['text'] is not None:
                batch['text'] = batch['text'].to(device)
                
            if batch['text'] is not None:
                x_recon_a, x_recon_t, mu, logvar = model(batch['audio'], batch['text'])
            else:
                x_recon_a, x_recon_t, mu, logvar = model(batch['audio'])
                
            loss, rec_a, rec_t, kld = loss_fn(batch, x_recon_a, x_recon_t, mu, logvar)
            
            metrics['loss'] += loss.item()
            metrics['rec_a'] += rec_a.item()
            metrics['rec_t'] += rec_t.item()
            metrics['kld'] += kld.item()

    for key in metrics:
        metrics[key] /= len(loader)
    return metrics

def plot_learning_curves(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.title('Total Loss')
    plt.legend()

    # Rec Audio
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_rec_a'], label='Train')
    plt.plot(epochs, history['val_rec_a'], label='Val')
    plt.title('Audio Reconstruction Loss')
    plt.legend()

    # Rec Text
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_rec_t'], label='Train')
    plt.plot(epochs, history['val_rec_t'], label='Val')
    plt.title('Text Reconstruction Loss')
    plt.legend()

    # KLD
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_kld'], label='Train')
    plt.plot(epochs, history['val_kld'], label='Val')
    plt.title('KLD Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_medium_task(model, train_loader, val_loader, epochs=50, device="cuda", name="model"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = HybridLoss()
    
    history = {
        'train_loss': [], 'train_rec_a': [], 'train_rec_t': [], 'train_kld': [],
        'val_loss': [], 'val_rec_a': [], 'val_rec_t': [], 'val_kld': []
    }
    
    print(f"Starting Training: {name} on {device}...")
    for epoch in range(epochs):
        train_m = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m = validate_epoch(model, val_loader, loss_fn, device)
        
        # Store history
        history['train_loss'].append(train_m['loss'])
        history['train_rec_a'].append(train_m['rec_a'])
        history['train_rec_t'].append(train_m['rec_t'])
        history['train_kld'].append(train_m['kld'])
        
        history['val_loss'].append(val_m['loss'])
        history['val_rec_a'].append(val_m['rec_a'])
        history['val_rec_t'].append(val_m['rec_t'])
        history['val_kld'].append(val_m['kld'])
        
        # Print metrics for every epoch
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  [Train] Loss: {train_m['loss']:.4f} | RecA: {train_m['rec_a']:.4f} | RecT: {train_m['rec_t']:.4f} | KLD: {train_m['kld']:.4f}")
        print(f"  [Val]   Loss: {val_m['loss']:.4f} | RecA: {val_m['rec_a']:.4f} | RecT: {val_m['rec_t']:.4f} | KLD: {val_m['kld']:.4f}")
            
    # Save curves
    plot_learning_curves(history, f"./results/medium_task/curves/{name}_loss_curves.png")
    
    return model, history