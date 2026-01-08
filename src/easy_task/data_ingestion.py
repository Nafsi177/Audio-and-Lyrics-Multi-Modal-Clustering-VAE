import os
import glob
import shutil
import kagglehub
from src.easy_task.config import raw_data_path
import subprocess

def download_and_setup():
    target_dir = raw_data_path

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    print("Downloading MIREX EMOTION Dataset...")
    download_path = kagglehub.dataset_download(
        "imsparsh/multimodal-mirex-emotion-dataset"
    )

    # Dataset structure fallback handling
    source_dir = os.path.join(download_path, "dataset")
    if not os.path.exists(source_dir):
        source_dir = download_path

    print(f"Copying files to {target_dir}...")

    items = os.listdir(source_dir)

    for item in items:
        src = os.path.join(source_dir, item)
        dst = os.path.join(target_dir, item)

        # Copy folders
        if os.path.isdir(src):
            if os.path.exists(dst):
                print(f"Folder '{item}' already exists. Skipping.")
                continue
            shutil.copytree(src, dst)
            print(f"Copied folder: {item}")

        # Copy files (e.g., .bat)
        elif os.path.isfile(src):
            if os.path.exists(dst):
                print(f"File '{item}' already exists. Skipping.")
                continue
            shutil.copy2(src, dst)
            print(f"Copied file: {item}")

    print("\nDataset setup complete!")
    print(f"Files are now located in: {target_dir}")


def get_raw_data():
    audio_files = glob.glob(os.path.join(raw_data_path, "Audio","*","*","*.mp3")) # data_path/Audio/cluster*/*/*.mp3
    lyrics_files = glob.glob(os.path.join(raw_data_path, "Lyrics","*","*","*.txt")) # data_path/cluster/cluster*/*.txt

    # Correct logical check
    if len(audio_files) == 0 or len(lyrics_files) == 0:
        download_and_setup()
        audio_files = glob.glob(os.path.join(raw_data_path, "Audio", "*.mp3"))
        lyrics_files = glob.glob(os.path.join(raw_data_path, "Lyrics", "*.txt"))

        if not audio_files or not lyrics_files:
            raise FileNotFoundError(
                "Could not find required files. Check dataset structure or paths or if the data download happen in src/easy_task/data_ingestion.py"
            )
        
        print("performing clusters given by the dataset creator. using the .bat script of the dataset.")
        audio_bat_path = f"{os.path.abspath(raw_data_path)}/split-by-categories-audio.bat"
        audio_dir = f"{os.path.abspath(raw_data_path)}/Audio"

        subprocess.run(
            ["cmd.exe", "/c", audio_bat_path],
            cwd=audio_dir,
            check=True
        )

        lyrics_bat_path = f"{os.path.abspath(raw_data_path)}/split-by-categories-lyrics.bat"
        lyrics_dir = f"{os.path.abspath(raw_data_path)}/Lyrics"

        subprocess.run(
            ["cmd.exe", "/c", lyrics_bat_path],
            cwd=lyrics_dir,
            check=True
        )
        audio_files = glob.glob(os.path.join(raw_data_path, "Audio","*","*","*.mp3")) # data_path/Audio/cluster*/*/*.mp3
        lyrics_files = glob.glob(os.path.join(raw_data_path, "Lyrics","*","*","*.txt")) # data_path/cluster/cluster*/*.txt

        if len(audio_files) == 0 or len(lyrics_files) == 0:
            raise FileNotFoundError(
                "Could not find required files even after downloading. Check src/easy-task/data_ingestion.py"
            )

    print("Total audio files found:", len(audio_files))
    print("Total lyrics files found:", len(lyrics_files))
    
    return audio_files, lyrics_files


if __name__ == "__main__":
    get_raw_data()
