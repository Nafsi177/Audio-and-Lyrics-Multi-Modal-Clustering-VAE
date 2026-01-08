import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add root to path
sys.path.append(os.getcwd())

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from src.medium_task.feature_engineering import get_multimodal_features
from src.medium_task.dataset import get_dataloaders
from src.medium_task.models import ConvVAE, HybridVAE
from src.medium_task.training_pipeline import train_medium_task

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define paths relative to the project root
BASE_RESULT_DIR = "./results/medium_task"
VIS_DIR = os.path.join(BASE_RESULT_DIR, "latent_visualization")
CURVE_DIR = os.path.join(BASE_RESULT_DIR, "curves")
CHECKPOINT_DIR = "./checkpoints/medium"

# --- Visualization Helper ---
def save_cluster_plot(embedding_2d, labels, title, filename):
    # Ensure filename is safe (remove parentheses and spaces)
    safe_filename = filename.replace("(", "").replace(")", "").replace(" ", "_")
    
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    cmap = 'tab10' if len(unique_labels) <= 10 else 'rainbow'
    
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1],
        c=labels, 
        cmap=cmap,
        s=20, 
        edgecolor='k', 
        alpha=0.7,
        linewidth=0.5
    )
    
    if -1 in unique_labels:
        plt.legend(*scatter.legend_elements(), title="Cluster ID")
    else:
        plt.colorbar(scatter, ticks=unique_labels, label="Cluster ID")
        
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SSE 2")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(VIS_DIR, safe_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {safe_filename}")

# --- Evaluation Helper ---
def evaluate_and_plot(features, labels_true, feature_name):
    results = []
    print(f"  > Computing t-SNE for {feature_name}...")
    
    if features.shape[1] > 50:
        pca_pre = PCA(n_components=50, random_state=SEED)
        feat_for_tsne = pca_pre.fit_transform(features)
    else:
        feat_for_tsne = features
        
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate='auto')
    embedding_2d = tsne.fit_transform(feat_for_tsne)

    K = len(set(labels_true))

    # KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(features)
    results.append({
        "Feature": feature_name, "Algorithm": "KMeans",
        "Silhouette": silhouette_score(features, km.labels_),
        "Davies-Bouldin": davies_bouldin_score(features, km.labels_),
        "ARI": adjusted_rand_score(labels_true, km.labels_),
        "Calinski-Harabasz": calinski_harabasz_score(features, km.labels_)
    })
    save_cluster_plot(embedding_2d, km.labels_, f"{feature_name} - KMeans", f"{feature_name}_KMeans.png")

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=K).fit_predict(features)
    results.append({
        "Feature": feature_name, "Algorithm": "Agglomerative",
        "Silhouette": silhouette_score(features, agg),
        "Davies-Bouldin": davies_bouldin_score(features, agg),
        "ARI": adjusted_rand_score(labels_true, agg),
        "Calinski-Harabasz": calinski_harabasz_score(features, agg)
    })
    save_cluster_plot(embedding_2d, agg, f"{feature_name} - Agglomerative", f"{feature_name}_Agglomerative.png")

    # DBSCAN
    db = DBSCAN(eps=5.0, min_samples=4).fit_predict(features)
    unique_db = len(set(db))
    if unique_db > 1:
        metrics = [silhouette_score(features, db), davies_bouldin_score(features, db), 
                   adjusted_rand_score(labels_true, db), calinski_harabasz_score(features, db)]
    else:
        metrics = [-1, -1, -1, -1]
        
    results.append({
        "Feature": feature_name, "Algorithm": "DBSCAN",
        "Silhouette": metrics[0], "Davies-Bouldin": metrics[1], "ARI": metrics[2], "Calinski-Harabasz": metrics[3]
    })
    save_cluster_plot(embedding_2d, db, f"{feature_name} - DBSCAN", f"{feature_name}_DBSCAN.png")
    
    return results

# --- Main Pipeline ---

def run_medium_task():
    # FORCE DIRECTORY CREATION AT START
    for d in [VIS_DIR, CURVE_DIR, CHECKPOINT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")

    print(">>> Step 1: Feature Engineering")
    audio_data, text_data, genres = get_multimodal_features()
    label_map = {g:i for i, g in enumerate(sorted(set(genres)))}
    labels_true = np.array([label_map[g] for g in genres])
    
    train_loader, val_loader = get_dataloaders(audio_data, text_data, batch_size=32)
    all_metrics = []

    # --- A. Baselines ---
    print("\n>>> Step 2: PCA Baselines")
    flat_audio = audio_data.reshape(audio_data.shape[0], -1)
    scaled_audio = StandardScaler().fit_transform(flat_audio)
    pca_audio = PCA(n_components=32, random_state=SEED).fit_transform(scaled_audio)
    all_metrics.extend(evaluate_and_plot(pca_audio, labels_true, "Baseline_PCA_Audio"))

    scaled_text = StandardScaler().fit_transform(text_data)
    hybrid_flat = np.hstack([scaled_audio, scaled_text])
    pca_hybrid = PCA(n_components=32, random_state=SEED).fit_transform(hybrid_flat)
    all_metrics.extend(evaluate_and_plot(pca_hybrid, labels_true, "Baseline_PCA_Hybrid"))

    # --- B. ConvVAE ---
    print("\n>>> Step 3: ConvVAE Training")
    conv_vae = ConvVAE(latent_dim=32)
    conv_vae, _ = train_medium_task(conv_vae, train_loader, val_loader, epochs=50, device=DEVICE, name="conv_vae")
    torch.save(conv_vae.state_dict(), os.path.join(CHECKPOINT_DIR, "conv_vae.pt"))
    
    conv_vae.eval()
    with torch.no_grad():
        latent_audio = []
        for i in range(0, len(audio_data), 32):
            b = torch.tensor(audio_data[i:i+32], dtype=torch.float32).to(DEVICE)
            _, _, mu, _ = conv_vae(b)
            latent_audio.append(mu.cpu().numpy())
    all_metrics.extend(evaluate_and_plot(np.vstack(latent_audio), labels_true, "ConvVAE_Audio"))

    # --- C. HybridVAE ---
    print("\n>>> Step 4: HybridVAE Training")
    hybrid_vae = HybridVAE(text_input_dim=128, latent_dim=32)
    hybrid_vae, _ = train_medium_task(hybrid_vae, train_loader, val_loader, epochs=50, device=DEVICE, name="hybrid_vae")
    torch.save(hybrid_vae.state_dict(), os.path.join(CHECKPOINT_DIR, "hybrid_vae.pt"))
    
    hybrid_vae.eval()
    with torch.no_grad():
        latent_hybrid = []
        for i in range(0, len(audio_data), 32):
            ba = torch.tensor(audio_data[i:i+32], dtype=torch.float32).to(DEVICE)
            bt = torch.tensor(text_data[i:i+32], dtype=torch.float32).to(DEVICE)
            _, _, mu, _ = hybrid_vae(ba, bt)
            latent_hybrid.append(mu.cpu().numpy())
    all_metrics.extend(evaluate_and_plot(np.vstack(latent_hybrid), labels_true, "HybridVAE_Full"))

    # --- Finalize ---
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(BASE_RESULT_DIR, "clustering_metric.csv"), index=False)
    print(f"\nSuccess! Metrics saved to {BASE_RESULT_DIR}/clustering_metric.csv")

if __name__ == "__main__":
    run_medium_task()