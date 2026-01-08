import os
import glob
from src.medium_task.config import raw_data_path

def get_paired_data():
    """
    Returns lists of paths for Audio and Lyrics that correspond to the same ID.
    Matches based on directory structure: .../Cluster X/Category/ID.ext
    """
    # 1. Map all audio files: Key = (Cluster, ID) -> Value = Path
    audio_map = {}
    audio_pattern = os.path.join(raw_data_path, "Audio", "*", "*", "*.mp3")
    for path in glob.glob(audio_pattern):
        # path example: .../Audio/Cluster 1/Boisterous/001.mp3
        parts = path.replace("\\", "/").split("/")
        filename = parts[-1]
        cluster = parts[-3] # This captures "Cluster 1", "Cluster 2" etc.
        file_id = os.path.splitext(filename)[0]
        
        # We store the Cluster ID as part of the unique key
        audio_map[(cluster, file_id)] = path

    # 2. Map all lyrics files
    lyrics_map = {}
    lyrics_pattern = os.path.join(raw_data_path, "Lyrics", "*", "*", "*.txt")
    for path in glob.glob(lyrics_pattern):
        parts = path.replace("\\", "/").split("/")
        filename = parts[-1]
        cluster = parts[-3]
        file_id = os.path.splitext(filename)[0]
        lyrics_map[(cluster, file_id)] = path

    # 3. Find Intersection
    common_keys = set(audio_map.keys()).intersection(set(lyrics_map.keys()))
    
    paired_audio = []
    paired_lyrics = []
    labels = []

    for key in common_keys:
        paired_audio.append(audio_map[key])
        paired_lyrics.append(lyrics_map[key])
        
        # FIX: Extract "Cluster X" as the label instead of the emotion category
        # Key is ((Cluster, ID)), so key[0] is the Cluster name
        labels.append(key[0]) 

    print(f"Found {len(audio_map)} audio, {len(lyrics_map)} lyrics.")
    print(f"Intersected Paired Dataset Size: {len(paired_audio)}")
    
    return paired_audio, paired_lyrics, labels

if __name__ == "__main__":
    get_paired_data()