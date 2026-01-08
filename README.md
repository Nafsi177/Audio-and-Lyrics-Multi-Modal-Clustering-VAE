# Audio-and-Lyrics-Multi-Modal-Clustering-VAE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Kaggle-blue)](https://www.kaggle.com/imsparsh/multimodal-mirex-emotion-dataset)

A reproducible research repository for multi-modal clustering using audio, lyrics, and MIDI data with VAE variants and PCA baselines. This project implements three progressively complex tasks (easy, medium, hard) for emotion clustering in music data.

## 📊 Overview

This project explores multi-modal representation learning for music emotion recognition using:
- **Audio features**: MFCCs, spectrograms, chroma features
- **Lyrical features**: TF-IDF, sentence embeddings
- **MIDI features**: Note sequences and musical patterns
- **Models**: VAE variants (ConvVAE, BetaVAE, CVAE) vs PCA baselines
- **Clustering**: K-Means, DBSCAN, Agglomerative clustering

## 🚀 Quick Start

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

### 4. Install Dependencies
```bash
pip install torch torchvision torchaudio librosa numpy pandas scikit-learn matplotlib tqdm faiss-cpu sentence-transformers kaggle
```

### 5. Download Dataset
```bash
# First, get your Kaggle API token from https://www.kaggle.com/account
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d imsparsh/multimodal-mirex-emotion-dataset --path data/raw --unzip
```

### 6. Run Experiments
```bash
# Easy task - Basic VAE vs PCA
python scripts/easy_task/easy_task.py

# Medium task - Advanced multi-modal features
python scripts/medium_task/medium_task.py

# Hard task - Complex VAE variants
python scripts/hard_task/hard_task.py
```

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

## 📊 Results Interpretation

Each experiment generates:

### Metrics Files (`clustering_metric.csv`)
- **Silhouette Score**: Measures cluster separation (-1 to 1, higher is better)
- **Calinski-Harabasz**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin**: Average similarity between clusters (lower is better)
- **ARI/NMI**: External validation if ground truth labels available

### Visualizations
- **t-SNE/UMAP plots**: 2D projections of latent spaces
- **Loss curves**: Training/validation loss over epochs
- **Reconstructions**: Original vs reconstructed samples
- **Cluster visualizations**: Latent space colored by cluster assignments

## ⚙️ Configuration

Edit configuration files in `src/*/config.py` to adjust:

```python
# Example configuration (easy_task/config.py)
DATA_DIR = "data/raw/multimodal-mirex-emotion-dataset"
RESULTS_DIR = "results/easy_task"
BATCH_SIZE = 32
LATENT_DIM = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
SEED = 42
```

## 🔧 Customization

### Add New Features
1. Extend `feature_engineering.py` with new feature extraction methods
2. Update `data_pipeline.py` to include new features
3. Adjust model input dimensions in `vae.py` or `models.py`

### Add New Models
1. Implement model in `models.py` or `vae.py`
2. Add corresponding loss function in `loss.py`
3. Update `training_pipeline.py` to support new model

### Change Clustering Algorithms
Modify the clustering section in `training_pipeline.py`:
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Change clustering algorithm
clusters = KMeans(n_clusters=8).fit_predict(latent_vectors)
# or
clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(latent_vectors)
```

## 🐛 Troubleshooting

### Common Issues

1. **Kaggle API Error**
   ```
   OSError: Could not find kaggle.json.
   ```
   **Solution**: Ensure `kaggle.json` is in `~/.kaggle/` with proper permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Out of Memory**
   **Solution**: Reduce batch size in config:
   ```python
   BATCH_SIZE = 16  # Instead of 32
   ```

3. **Missing Dependencies**
   **Solution**: Install missing packages:
   ```bash
   pip install [missing-package-name]
   ```

4. **Slow Feature Extraction**
   **Solution**: Implement caching in `feature_engineering.py`:
   ```python
   import joblib
   cache_path = f"cache/{feature_name}.pkl"
   if os.path.exists(cache_path):
       features = joblib.load(cache_path)
   else:
       features = extract_features(data)
       joblib.dump(features, cache_path)
   ```

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
└── metadata.csv  # (if available)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{multimodal_mirex_emotion_dataset,
  author = {Sparsh, I.},
  title = {Multi-modal MIREX Emotion Dataset},
  year = {2023},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/imsparsh/multimodal-mirex-emotion-dataset}}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Provide detailed description and error logs
3. Include your environment details

## 🚀 Advanced Usage

### Using GPU Acceleration
```python
# In training_pipeline.py, ensure:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### Reproducibility
Set random seeds for reproducibility:
```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

### Extending to New Datasets
1. Update `data_ingestion.py` to handle new file formats
2. Modify `preprocess.py` for dataset-specific preprocessing
3. Adjust paths in `config.py`

---
**Note**: This project is for research purposes. Always respect copyright when working with audio data.