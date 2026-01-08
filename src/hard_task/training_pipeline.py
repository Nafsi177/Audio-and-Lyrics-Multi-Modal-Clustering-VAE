import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.hard_task.loss import HardTaskLoss
from src.hard_task.models import ConditionalVAE, SimpleAutoencoder

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    metrics = {'loss': 0, 'rec_a': 0, 'rec_t': 0, 'kld': 0}
    
    for batch in loader:
        batch['audio'] = batch['audio'].to(device)
        batch['text'] = batch['text'].to(device)
        if 'label_vec' in batch:
            batch['label_vec'] = batch['label_vec'].to(device)
            
        optimizer.zero_grad()
        
        # Forward Pass handling for different models
        if isinstance(model, ConditionalVAE):
            x_recon_a, x_recon_t, mu, logvar, _ = model(batch['audio'], batch['text'], batch['label_vec'])
        elif isinstance(model, SimpleAutoencoder):
            # AE returns (rec_a, rec_t, z) -> No KLD
            x_recon_a, x_recon_t, _ = model(batch['audio'], batch['text'])
            mu, logvar = None, None
        else:
            # Beta-VAE / HybridVAE
            x_recon_a, x_recon_t, mu, logvar = model(batch['audio'], batch['text'])
            
        loss, rec_a, rec_t, kld = loss_fn(batch, x_recon_a, x_recon_t, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        metrics['loss'] += loss.item()
        metrics['rec_a'] += rec_a.item()
        metrics['rec_t'] += rec_t.item()
        metrics['kld'] += kld.item()
        
    for key in metrics:
        metrics[key] /= len(loader)
        
    return metrics

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    metrics = {'loss': 0, 'rec_a': 0, 'rec_t': 0, 'kld': 0}
    
    with torch.no_grad():
        for batch in loader:
            batch['audio'] = batch['audio'].to(device)
            batch['text'] = batch['text'].to(device)
            if 'label_vec' in batch:
                batch['label_vec'] = batch['label_vec'].to(device)
            
            if isinstance(model, ConditionalVAE):
                x_recon_a, x_recon_t, mu, logvar, _ = model(batch['audio'], batch['text'], batch['label_vec'])
            elif isinstance(model, SimpleAutoencoder):
                x_recon_a, x_recon_t, _ = model(batch['audio'], batch['text'])
                mu, logvar = None, None
            else:
                x_recon_a, x_recon_t, mu, logvar = model(batch['audio'], batch['text'])
                
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
    
    # 1. Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.title('Total Loss')
    plt.legend()
    
    # 2. Audio Reconstruction
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_rec_a'], label='Train')
    plt.plot(epochs, history['val_rec_a'], label='Val')
    plt.title('Audio Recon Loss')
    plt.legend()
    
    # 3. Text Reconstruction
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_rec_t'], label='Train')
    plt.plot(epochs, history['val_rec_t'], label='Val')
    plt.title('Text Recon Loss')
    plt.legend()
    
    # 4. KLD
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_kld'], label='Train')
    plt.plot(epochs, history['val_kld'], label='Val')
    plt.title('KLD Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_hard_task(model, train_loader, val_loader, epochs, beta_val, device, name, save_dir):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Initialize Loss with specific Beta value
    loss_fn = HardTaskLoss(beta=beta_val)
    
    history = {
        'train_loss': [], 'train_rec_a': [], 'train_rec_t': [], 'train_kld': [],
        'val_loss': [], 'val_rec_a': [], 'val_rec_t': [], 'val_kld': []
    }
    
    print(f"Starting Training: {name} (Beta={beta_val}) on {device}...")
    
    for epoch in range(epochs):
        train_m = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m = validate_epoch(model, val_loader, loss_fn, device)
        
        # Append history
        history['train_loss'].append(train_m['loss'])
        history['train_rec_a'].append(train_m['rec_a'])
        history['train_rec_t'].append(train_m['rec_t'])
        history['train_kld'].append(train_m['kld'])
        
        history['val_loss'].append(val_m['loss'])
        history['val_rec_a'].append(val_m['rec_a'])
        history['val_rec_t'].append(val_m['rec_t'])
        history['val_kld'].append(val_m['kld'])
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  [Train] Loss: {train_m['loss']:.2f} | RecA: {train_m['rec_a']:.2f} | KLD: {train_m['kld']:.2f}")
            print(f"  [Val]   Loss: {val_m['loss']:.2f} | RecA: {val_m['rec_a']:.2f} | KLD: {val_m['kld']:.2f}")

    # Save Curves
    curve_path = os.path.join(save_dir, "curves", f"{name}_curves.png")
    plot_learning_curves(history, curve_path)
    
    return model, history