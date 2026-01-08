import torch
import torch.nn as nn
import torch.nn.functional as F

class HardTaskLoss(nn.Module):
    def __init__(self, beta=1.0, alpha=1.0):
        super().__init__()
        self.beta = beta   # Controls disentanglement
        self.alpha = alpha # Controls audio weight
        
    def forward(self, batch, recon_audio, recon_text, mu, logvar):
        x_audio = batch['audio']
        x_text = batch['text']

        # 1. Reconstruction Loss (MSE)
        loss_audio = F.mse_loss(recon_audio, x_audio, reduction='sum')
        loss_text = F.mse_loss(recon_text, x_text, reduction='sum')
        
        # 2. KL Divergence
        # If mu/logvar are None (e.g. Autoencoder), KLD is 0
        if mu is None or logvar is None:
            kld = torch.tensor(0.0, device=x_audio.device)
        else:
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total
        total_loss = (self.alpha * loss_audio) + loss_text + (self.beta * kld)
        
        # Normalize by batch size for logging readability
        batch_size = x_audio.size(0)
        
        return total_loss / batch_size, loss_audio / batch_size, loss_text / batch_size, kld / batch_size