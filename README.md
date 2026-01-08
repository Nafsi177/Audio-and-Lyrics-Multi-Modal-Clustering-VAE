# Audio-and-Lyrics-Multi-Modal-Clustering-VAE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Kaggle-blue)](https://www.kaggle.com/imsparsh/multimodal-mirex-emotion-dataset)

A reproducible research repository for multi-modal clustering using audio, lyrics, and MIDI data with VAE variants and PCA baselines. This project implements three progressively complex tasks (easy, medium, hard) for clustering in music data (audio+text) using different varients of VAE.

## 📊 Overview

This project explores multi-modal representation learning for music emotion recognition using:
- **Audio features**: MFCCs, spectrograms, chroma features
- **Lyrical features**: TF-IDF, sentence embeddings
- **MIDI features**: Note sequences and musical patterns
- **Models**: VAE variants (ConvVAE, BetaVAE, CVAE) vs PCA baselines
- **Clustering**: K-Means, DBSCAN, Agglomerative clustering

##  Reproducing experiment results...

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Audio-and-Lyrics-Multi-Modal-Clustering-VAE.git
cd Audio-and-Lyrics-Multi-Modal-Clustering-VAE
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install Package
```bash
pip install -e .
```

### 4. Reproduce Experiments result
```bash
# Easy task - Basic VAE vs PCA
python scripts/easy_task/easy_task.py

# Medium task - Advanced multi-modal features
python scripts/medium_task/medium_task.py

# Hard task - Complex VAE variants
python scripts/hard_task/hard_task.py
```
**Note**: Make sure to run in the following sequence: easy_task.py --> medium_task.py -->hard_task.py 

## 📁 Project Structure

```
Audio-and-Lyrics-Multi-Modal-Clustering-VAE/
├── src/                          # Source code
│   ├── easy_task/               # Basic VAE implementation
│   │   ├── config.py           # Hyperparameters & paths
│   │   ├── data_ingestion.py   # Load audio/lyrics/midi
│   │   ├── data_pipeline.py    # Orchestration pipeline
│   │   ├── feature_engineering.py # Feature extraction
│   │   ├── loss.py             # VAE loss functions
│   │   ├── preprocess.py       # Audio/lyrics preprocessing
│   │   ├── training_pipeline.py # Training loop
│   │   └── vae.py              # VAE model definition
│   ├── medium_task/            # Advanced multi-modal
│   │   ├── config.py
│   │   ├── data_ingestion.py
│   │   ├── dataset.py
│   │   ├── feature_engineering.py
│   │   ├── loss.py
│   │   ├── models.py
│   │   └── training_pipeline.py
│   └── hard_task/              # Complex VAE variants
│       ├── config.py
│       ├── dataset.py
│       ├── loss.py
│       ├── models.py
│       └── training_pipeline.py
├── scripts/                     # Entry points
│   ├── easy_task/easy_task.py
│   ├── medium_task/medium_task.py
│   └── hard_task/hard_task.py
├── results/                     # Generated outputs
│   ├── easy_task/
│   │   ├── clustering_metric.csv
│   │   ├── latent_visualization/
│   │   └── vae_traintime_metric.png
│   ├── medium_task/
│   │   ├── clustering_metric.csv
│   │   ├── curves/
│   │   └── latent_visualization/
│   └── hard_task/
│       ├── curves/
│       ├── hard_task_metrics.csv
│       └── visualizations/
├── data/                        # Dataset (not in repo)
│   └── raw/
│       └── multimodal-mirex-emotion-dataset/
│           ├── audio/
│           ├── lyrics/
│           ├── midi/
│           └── metadata.csv
└── README.md                    # This file
```

## 📈 Experiments

### Easy Task
- **Objective**: Compare vanilla VAE vs PCA for audio feature clustering
- **Features**: MFCCs, spectrograms, basic lyric embeddings
- **Models**: Vanilla VAE, PCA baseline
- **Output**: Basic clustering metrics and latent visualizations

### Medium Task
- **Objective**: Multi-modal clustering with advanced features
- **Features**: Chroma features, mel-spectrograms, combined embeddings
- **Models**: ConvVAE, HybridVAE (audio + lyrics)
- **Output**: Comparison of different clustering algorithms

### Hard Task
- **Objective**: Advanced VAE variants for emotion clustering
- **Models**: Autoencoder, BetaVAE, CVAE
- **Features**: All modalities with advanced fusion
- **Output**: Comprehensive metrics and reconstruction visualizations

### Visualizations
- **t-SNE plots**: 2D projections of latent spaces
- **Loss curves**: Training/validation loss over epochs
- **Reconstructions**: Original vs reconstructed samples
- **Cluster visualizations**: Latent space colored by cluster assignments

## 📚 Dataset Information

**Multi-modal MIREX Emotion Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/imsparsh/multimodal-mirex-emotion-dataset)
- **Contents**: Audio clips, lyrics, and MIDI files labeled with emotions
- **Emotions**: Happy, Sad, Angry, Relaxed, etc.
- **Format**: MP3 audio, TXT lyrics, MIDI files
- **Size**: ~1000 multi-modal samples

### Expected Data Structure
```
data/raw/multimodal-mirex-emotion-dataset/
├── audio/
│   ├── song1.mp3
│   ├── song2.mp3
│   └── ...
├── lyrics/
│   ├── song1.txt
│   ├── song2.txt
│   └── ...
├── midi/
│   ├── song1.mid
│   ├── song2.mid
│   └── ...
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Provide detailed description and error logs
3. Include your environment details

---
**Note**: This project is for research purposes. 