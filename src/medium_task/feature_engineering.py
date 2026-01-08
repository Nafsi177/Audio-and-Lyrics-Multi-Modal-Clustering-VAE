import librosa
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from src.medium_task.config import audio_configs, text_configs, image_dims
from src.medium_task.data_ingestion import get_paired_data
from tqdm import tqdm
import cv2

def extract_spectrogram(path):
    """
    Extracts Mel Spectrogram and resizes to (64, 128).
    """
    y, sr = librosa.load(path, sr=audio_configs.sample_rate, duration=audio_configs.duration)
    
    # Pad if too short
    target_len = int(audio_configs.sample_rate * audio_configs.duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Mel Spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, 
        n_mels=audio_configs.n_mels, 
        n_fft=audio_configs.n_fft, 
        hop_length=audio_configs.hop_length
    )
    S_log = librosa.power_to_db(S, ref=np.max)

    # Resize to fixed dimensions (64, 128) for ConvNet
    # OpenCV expects (Width, Height)
    S_resized = cv2.resize(S_log, (image_dims[2], image_dims[1])) 
    
    # Add channel dimension (1, 64, 128)
    S_resized = S_resized[np.newaxis, ...] 
    
    # Min-Max Scaling to [0, 1] for BCE Loss / Stability
    s_min, s_max = S_resized.min(), S_resized.max()
    if s_max - s_min > 0:
        S_resized = (S_resized - s_min) / (s_max - s_min)
        
    return S_resized.astype(np.float32)

def extract_text_features(lyrics_paths):
    """
    Reads text files and returns TF-IDF vectors.
    """
    corpus = []
    valid_indices = []
    
    print("Reading Lyrics files...")
    for idx, path in enumerate(lyrics_paths):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().replace('\n', ' ')
                if len(text.strip()) > 0:
                    corpus.append(text)
                    valid_indices.append(idx)
                else:
                    # Handle empty files by adding placeholder to maintain alignment 
                    # (or we could drop, but logic is easier if we keep length)
                    corpus.append("unknown") 
                    valid_indices.append(idx)
        except Exception:
            corpus.append("unknown")
            valid_indices.append(idx)

    vectorizer = TfidfVectorizer(max_features=text_configs.max_features, stop_words='english')
    X_text = vectorizer.fit_transform(corpus).toarray()
    
    return X_text.astype(np.float32)

def get_multimodal_features():
    audio_paths, lyrics_paths, genres = get_paired_data()
    
    # Process Audio
    print("Extracting Audio Spectrograms...")
    audio_feats = []
    for p in tqdm(audio_paths):
        audio_feats.append(extract_spectrogram(p))
    audio_feats = np.stack(audio_feats) # (N, 1, 64, 128)
    
    # Process Text
    print("Extracting Lyrics TF-IDF...")
    text_feats = extract_text_features(lyrics_paths) # (N, 128)
    
    return audio_feats, text_feats, np.array(genres)

if __name__ == "__main__":
    a, t, g = get_multimodal_features()
    print("Audio shape:", a.shape)
    print("Text shape:", t.shape)