import torch
import torch.nn as nn

class ELBO_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat, mu, logvar):
        # Reconstruction: MSE over all elements
        recon = torch.mean((x - x_hat) ** 2)
        # KL divergence: mean over all elements
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + kld
        return loss, recon.detach(), kld.detach()


# class ELBO_Loss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, x_hat, mu, logvar):
#         # Reconstruction Loss (MSE)
#         # Using mean ensures the loss doesn't scale wildly with image size
#         recon = torch.mean((x - x_hat) ** 2)

#         # KL Divergence
#         # Sum over the latent dimension (dim=1), then Mean over the batch
#         # This is more stable for training latent representations
#         kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
#         # Total Loss
#         loss = recon + kld
        
#         return loss, recon, kld
