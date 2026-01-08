import torch
import torch.nn as nn

class BasicVAE(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, latent_dim=8):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2,hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4,hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterization(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar
    

if __name__=="__main__":
    INPUT_DIM = 40
    HIDDEN_DIM = 128
    LATENT_DIM = 8
    model=BasicVAE(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)
    print(model)

