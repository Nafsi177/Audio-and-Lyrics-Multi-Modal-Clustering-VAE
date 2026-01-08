from collections import namedtuple
from src.medium_task.config import audio_configs, text_configs, image_dims, raw_data_path

# Hard Task Specific Configs
hard_configs = namedtuple("hard_configs", 
    ['latent_dim', 'beta_value', 'cvae_hidden', 'batch_size', 'epochs']
)(
    latent_dim=32,
    beta_value=4.0,   # Stronger disentanglement
    cvae_hidden=128,
    batch_size=32,
    epochs=50
)