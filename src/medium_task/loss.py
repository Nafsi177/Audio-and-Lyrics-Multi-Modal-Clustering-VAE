import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha # Weight for Audio
        self.beta = beta   # Weight for Text
        
    def forward(self, batch, recon_audio, recon_text, mu, logvar):
        x_audio = batch['audio']
        x_text = batch['text'] if 'text' in batch else None

        # 1. Audio Recon (MSE or BCE) - using MSE here
        loss_audio = F.mse_loss(recon_audio, x_audio, reduction='sum')
        
        # 2. Text Recon (MSE) - only if Hybrid
        loss_text = torch.tensor(0.0, device=x_audio.device)
        if recon_text is not None and x_text is not None:
            loss_text = F.mse_loss(recon_text, x_text, reduction='sum')

        # 3. KLD
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = (self.alpha * loss_audio) + (self.beta * loss_text) + kld
        
        return total_loss, loss_audio, loss_text, kld