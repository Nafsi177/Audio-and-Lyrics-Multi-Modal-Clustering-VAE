import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, channel, height, width):
        super().__init__()
        self.c, self.h, self.w = channel, height, width
    def forward(self, input):
        return input.view(input.size(0), self.c, self.h, self.w)

# --- Components ---

class AudioEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Input (1, 64, 128)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # -> (32, 32, 64)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 16, 32)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 8, 16)
            nn.BatchNorm2d(128), nn.ReLU(),
            Flatten()
        )
        self.flat_dim = 128 * 8 * 16 # 16384

    def forward(self, x):
        return self.net(x)

class AudioDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 16)
        self.unflat = UnFlatten(128, 8, 16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # (64, 16, 32)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 32, 64)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # (1, 64, 128)
            nn.Sigmoid() # Output 0-1 (we scaled data to 0-1)
        )

    def forward(self, z):
        h = self.fc(z)
        h = self.unflat(h)
        return self.net(h)

# --- ConvVAE (Audio Only) ---

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = AudioEncoder()
        self.mu = nn.Linear(self.encoder.flat_dim, latent_dim)
        self.logvar = nn.Linear(self.encoder.flat_dim, latent_dim)
        self.decoder = AudioDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text=None): # x_text ignored here
        h = self.encoder(x_audio)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, None, mu, logvar # Return None for text recon

# --- HybridVAE (Audio + Lyrics) ---

class HybridVAE(nn.Module):
    def __init__(self, text_input_dim=128, latent_dim=32):
        super().__init__()
        # Audio Enc
        self.audio_enc = AudioEncoder() # Output 16384
        
        # Text Enc
        self.text_enc = nn.Sequential(
            nn.Linear(text_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Fusion
        fusion_dim = self.audio_enc.flat_dim + 128
        self.mu = nn.Linear(fusion_dim, latent_dim)
        self.logvar = nn.Linear(fusion_dim, latent_dim)
        
        # Audio Dec
        self.audio_dec = AudioDecoder(latent_dim)
        
        # Text Dec
        self.text_dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, text_input_dim) # Output logits or similar (MSE used generally)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text):
        # Encode Audio
        h_a = self.audio_enc(x_audio)
        # Encode Text
        h_t = self.text_enc(x_text)
        
        # Concatenate
        h = torch.cat([h_a, h_t], dim=1)
        
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_audio = self.audio_dec(z)
        recon_text = self.text_dec(z)
        
        return recon_audio, recon_text, mu, logvar