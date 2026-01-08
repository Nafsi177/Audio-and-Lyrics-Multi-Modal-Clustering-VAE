```md
# Audio-and-Lyrics-Multi-Modal-Clustering-VAE

A fully reproducible research repository for **multi-modal emotion clustering** using **audio, lyrics, and MIDI** data.  
This project evaluates **VAE-based latent representations** against **classical PCA baselines** across **easy, medium, and hard experimental tasks**, using multiple clustering algorithms.

---

## Dataset

**Multi-modal MIREX Emotion Dataset (Kaggle)**  
Link: https://www.kaggle.com/datasets/imsparsh/multimodal-mirex-emotion-dataset/data

The dataset contains:
- 🎵 Audio files  
- 📝 Lyrics  
- 🎹 MIDI data  
- 😃 Emotion labels  

All experiments in this repository are based on this dataset.

---

## Repository Structure

```

Audio-and-Lyrics-Multi-Modal-Clustering-VAE/
│
├── src/
│   ├── **init**.py
│   │
│   ├── easy_task/
│   │   ├── **init**.py
│   │   ├── config.py
│   │   ├── data_ingestion.py
│   │   ├── data_pipeline.py
│   │   ├── feature_engineering.py
│   │   ├── preprocess.py
│   │   ├── loss.py
│   │   ├── vae.py
│   │   └── training_pipeline.py
│   │
│   ├── medium_task/
│   │   ├── config.py
│   │   ├── data_ingestion.py
│   │   ├── dataset.py
│   │   ├── feature_engineering.py
│   │   ├── loss.py
│   │   ├── models.py
│   │   └── training_pipeline.py
│   │
│   └── hard_task/
│       ├── config.py
│       ├── dataset.py
│       ├── loss.py
│       ├── models.py
│       └── training_pipeline.py
│
├── scripts/
│   ├── easy_task/
│   │   └── easy_task.py
│   ├── medium_task/
│   │   └── medium_task.py
│   └── hard_task/
│       └── hard_task.py
│
└── results/
├── easy_task/
├── medium_task/
└── hard_task/

````

---

## Purpose of Each Component

### `src/` (Core library)
Contains **all reusable components** for data loading, preprocessing, feature extraction, modeling, training, and evaluation.

---

### Easy Task (`src/easy_task/`)
Designed as a **baseline and sanity-check pipeline**.

- `config.py`  
  Central configuration (paths, hyperparameters, seeds).

- `data_ingestion.py`  
  Loads raw audio/lyrics/MIDI files and metadata.

- `preprocess.py`  
  Audio resampling, trimming, lyric cleaning.

- `feature_engineering.py`  
  Feature extraction (e.g., MFCCs, spectrograms, text embeddings).

- `vae.py`  
  Vanilla / convolutional VAE architecture.

- `loss.py`  
  Reconstruction + KL divergence (ELBO).

- `data_pipeline.py`  
  End-to-end data flow orchestration.

- `training_pipeline.py`  
  Training loop, latent extraction, clustering, metrics, and visualization.

---

### Medium Task (`src/medium_task/`)
Introduces **hybrid (audio + lyrics) models** and multiple clustering strategies.

- `dataset.py`  
  PyTorch Dataset and DataLoader logic.

- `models.py`  
  ConvVAE and HybridVAE architectures.

- `feature_engineering.py`  
  Advanced multimodal feature fusion.

- `loss.py`  
  Hybrid loss functions.

- `training_pipeline.py`  
  Model training, clustering (KMeans, DBSCAN, Agglomerative), evaluation.

---

### Hard Task (`src/hard_task/`)
Focuses on **complex latent modeling** and **comparative baselines**.

- `models.py`  
  Autoencoder, BetaVAE, CVAE implementations.

- `loss.py`  
  β-VAE and conditional VAE losses.

- `dataset.py`  
  Complex multimodal batching.

- `training_pipeline.py`  
  Training, reconstruction visualization, clustering comparison.

---

## Scripts (`scripts/`)

Each script is a **reproducible experiment entry point**.

- `scripts/easy_task/easy_task.py`
- `scripts/medium_task/medium_task.py`
- `scripts/hard_task/hard_task.py`

### Important
> These scripts **regenerate all results** by calling components from `src/`.  
> Outputs are automatically saved to `results/<task_name>/`.

---

## Results Overview

### `results/easy_task/`
- `clustering_metric.csv`  
  Silhouette, Calinski-Harabasz, Davies-Bouldin scores.

- `latent_visualization/`
  - `VAE_kmeans(tsne).png`
  - `pca_kmeans(tsne).png`

- `vae_traintime_metric.png`  
  Training loss and runtime comparison.

---

### `results/medium_task/`
- `clustering_metric.csv`
- `curves/`
  - `conv_vae_loss_curves.png`
  - `hybrid_vae_loss_curves.png`
- `latent_visualization/`
  PCA vs ConvVAE vs HybridVAE with KMeans / DBSCAN / Agglomerative clustering.

---

### `results/hard_task/`
- `hard_task_metrics.csv`
- `curves/`
  - Autoencoder, BetaVAE, CVAE training curves.
- `visualizations/`
  - Reconstructions and t-SNE latent plots for all models.

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Audio-and-Lyrics-Multi-Modal-Clustering-VAE
````

### 2. Create virtual environment (recommended)

```bash
python -m venv .venv
```

Activate:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install the project (editable mode)

```bash
pip install -e .
```

---

## Dataset Download

### Using Kaggle API

```bash
pip install kaggle
```

1. Create API token from Kaggle account.
2. Place `kaggle.json` in:

   * `~/.kaggle/` (Linux/macOS)
   * `%USERPROFILE%\.kaggle\` (Windows)

```bash
kaggle datasets download -d imsparsh/multimodal-mirex-emotion-dataset --path data/raw --unzip
```

Expected structure:

```
data/raw/multimodal-mirex-emotion-dataset/
├── audio/
├── lyrics/
├── midi/
└── metadata.csv
```

Update dataset paths if needed in:

* `src/easy_task/config.py`
* `src/medium_task/config.py`
* `src/hard_task/config.py`

---

## Reproducing Results

Run each experiment script:

```bash
python scripts/easy_task/easy_task.py
python scripts/medium_task/medium_task.py
python scripts/hard_task/hard_task.py
```

Each command will:

* Preprocess data
* Extract features
* Train models
* Perform clustering
* Save metrics and visualizations to `results/`

---

## Reproducibility Notes

* Set random seeds in `training_pipeline.py` for deterministic results.
* Adjust batch size or latent dimension if GPU memory is limited.
* Use CPU by default; CUDA is auto-detected if available.

---

## License & Attribution

* Dataset credit: Kaggle – Multi-modal MIREX Emotion Dataset
* Use for academic and research purposes only.
* Please cite the dataset and this repository if used in publications.

---

## Future Improvements

* Add `requirements.txt`
* Add Dockerfile for full reproducibility
* Add automated experiment runner (Makefile)
* Add evaluation notebooks

---

**Author**: Jannatul Feardous Nafsi
**Research Area**: Multi-modal Representation Learning, VAE-based Clustering

```
```
