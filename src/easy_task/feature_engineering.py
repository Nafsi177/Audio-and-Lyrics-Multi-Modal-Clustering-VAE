from src.easy_task.config import mfcc_configs
import numpy as np
import librosa
from src.easy_task.data_ingestion import get_raw_data
from tqdm import tqdm
def extract_mfcc_features(path, sr=mfcc_configs.sample_rate, duration=mfcc_configs.duration, n_mfcc=mfcc_configs.n_mfcc):
    y, _sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std], axis=0) 
    return feat.astype(np.float32)

def parse_genre_from_path(path):
    parts = path.replace("\\", "/").split("/")
    return parts[-3]

def get_features():
    raw_wav_audio_files,_= get_raw_data()
    features = []
    genres = []
    paths = []
    for p in tqdm(raw_wav_audio_files):
        try:
            x = extract_mfcc_features(p)
            features.append(x)
            genres.append(parse_genre_from_path(p))
            paths.append(p)
        except Exception as e:
            # skip problematic files
            print("Skipping:", p, "Error:", str(e))
    features = np.vstack(features)
    genres = np.array(genres)
    print("Feature matrix shape:", features.shape)
    print("Unique genres:", sorted(set(genres)))
    return features,genres,paths


if __name__=="__main__":
    features,genres,paths=get_features()
    