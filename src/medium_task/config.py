from collections import namedtuple

# Audio Configs for Mel Spectrogram
audio_configs = namedtuple("audio_configs", 
    ['sample_rate', 'duration', 'n_mels', 'n_fft', 'hop_length']
)(
    sample_rate=22050, 
    duration=30.0, 
    n_mels=64,       # Height of spectrogram
    n_fft=2048, 
    hop_length=512
)

# Text Configs
text_configs = namedtuple("text_configs", ['max_features'])(max_features=128)

# Input dimension for ConvVAE (Time steps calculation: ~1292 for 30s at 22k sr/512 hop)
# We will resize spectrograms to fixed (64, 128) for easy Conv operations
image_dims = (1, 64, 128)  # Channel, Height (n_mels), Width (Time)

raw_data_path = "./data"