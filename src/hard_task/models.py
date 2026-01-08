import torch
import torch.nn as nn
import torch.nn.functional as F
from src.medium_task.models import AudioEncoder, AudioDecoder, Flatten, UnFlatten

# --- 1. Baseline Autoencoder ---
class SimpleAutoencoder(nn.Module):
    def __init__(self, text_input_dim=128, latent_dim=32):
        super().__init__()
        self.audio_enc = AudioEncoder()
        self.text_enc = nn.Linear(text_input_dim, 128)
        
        # Bottleneck
        fusion_dim = self.audio_enc.flat_dim + 128
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder Preparation
        self.to_audio_shape = nn.Linear(latent_dim, 128 * 8 * 16)
        self.unflat = UnFlatten(128, 8, 16)
        self.audio_dec_net = AudioDecoder(latent_dim).net # Reuse Conv parts
        
        self.text_dec = nn.Linear(latent_dim, text_input_dim)

    def forward(self, x_audio, x_text, label=None): # Label argument for compatibility
        # Encode
        h_a = self.audio_enc(x_audio)
        h_t = F.relu(self.text_enc(x_text))
        h = torch.cat([h_a, h_t], dim=1)
        
        z = self.bottleneck(h)
        
        # Decode
        h_rec_a = self.to_audio_shape(z)
        h_rec_a = self.unflat(h_rec_a)
        rec_audio = self.audio_dec_net(h_rec_a)
        
        rec_text = self.text_dec(z)
        
        return rec_audio, rec_text, z

# --- 2. Conditional VAE (CVAE) ---
class ConditionalVAE(nn.Module):
    def __init__(self, n_classes, text_input_dim=128, latent_dim=32):
        super().__init__()
        self.n_classes = n_classes
        
        self.audio_enc = AudioEncoder() 
        self.text_enc = nn.Sequential(
            nn.Linear(text_input_dim, 128),
            nn.ReLU()
        )
        
        # Fusion: Audio + Text + Label
        fusion_input_dim = self.audio_enc.flat_dim + 128 + n_classes
        
        self.fc_mu = nn.Linear(fusion_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_input_dim, latent_dim)
        
        # Decoder Input: Latent + Label
        decode_input_dim = latent_dim + n_classes
        
        # Audio Decoder Bridge
        self.fc_audio_dec = nn.Linear(decode_input_dim, 128 * 8 * 16)
        self.unflat = UnFlatten(128, 8, 16)
        self.audio_net = AudioDecoder(32).net 
        
        # Text Decoder
        self.text_dec = nn.Sequential(
            nn.Linear(decode_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, text_input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text, y_label):
        # Encode
        h_a = self.audio_enc(x_audio)
        h_t = self.text_enc(x_text)
        
        # Concat condition
        h = torch.cat([h_a, h_t, y_label], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decode (Conditioned)
        z_cond = torch.cat([z, y_label], dim=1)
        
        h_d_a = self.fc_audio_dec(z_cond)
        h_d_a = self.unflat(h_d_a)
        rec_audio = self.audio_net(h_d_a)
        
        rec_text = self.text_dec(z_cond)
        
        return rec_audio, rec_text, mu, logvar, z