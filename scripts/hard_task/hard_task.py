import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path setup
sys.path.append(os.getcwd())

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Project Imports
from src.medium_task.feature_engineering import get_multimodal_features
from src.medium_task.models import HybridVAE 
from src.hard_task.dataset import get_hard_dataloaders
from src.hard_task.models import ConditionalVAE, SimpleAutoencoder
from src.hard_task.training_pipeline import train_hard_task
from src.hard_task.config import hard_configs

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = "./results/hard_task"
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
CURVE_DIR = os.path.join(BASE_DIR, "curves")
CHECKPOINT_DIR = "./checkpoints/hard"

# --- Metrics Helper ---
def cluster_purity(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(contingency_matrix.max(axis=0)) / np.sum(contingency_matrix.values)

def evaluate_clustering(features, labels_true, algo_name):
    k = len(set(labels_true))
    if k < 2 or len(features) == 0: return {}

    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels_pred = kmeans.fit_predict(features)
    
    return {
        "Method": algo_name,
        "Silhouette": silhouette_score(features, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "ARI": adjusted_rand_score(labels_true, labels_pred),
        "Purity": cluster_purity(labels_true, labels_pred),
        "labels_pred": labels_pred # Keep for plotting
    }

# --- Plotting Helper (Styled like easy_task.py) ---
def save_latent_plot(features, pred_labels, title, filename):
    if features.shape[1] > 50:
        features = PCA(n_components=50, random_state=SEED).fit_transform(features)
    
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb[:, 0], 
        emb[:, 1],
        c=pred_labels, 
        cmap=plt.get_cmap("rainbow"),
        s=30, 
        edgecolor='k', 
        alpha=0.8
    )
    
    plt.title(title)
    # Match colorbar style: one tick per cluster
    unique_labels = sorted(list(set(pred_labels)))
    plt.colorbar(scatter, ticks=unique_labels, label="Cluster")
    
    plt.savefig(os.path.join(VIS_DIR, filename))
    plt.close()

def save_reconstruction_plot(model, loader, device, name, is_cvae=False):
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    a = batch['audio'].to(device)
    t = batch['text'].to(device)
    l = batch['label_vec'].to(device) if is_cvae else None
    
    with torch.no_grad():
        if is_cvae:
            rec_a, _, _, _, _ = model(a, t, l)
        else:
            out = model(a, t)
            rec_a = out[0]

    orig = a.cpu().numpy()
    recon = rec_a.cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        if i >= len(orig): break
        # Using magma as requested in your original hard_task script
        axes[0, i].imshow(orig[i, 0], aspect='auto', origin='lower', cmap='magma')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon[i, 0], aspect='auto', origin='lower', cmap='magma')
        axes[1, i].set_title(f"Recon {name}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"recon_{name}.png"))
    plt.close()

# --- Main ---
def run_hard_task():
    for d in [VIS_DIR, CURVE_DIR, CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

    print(">>> 1. Loading Data...")
    audio, text, genres = get_multimodal_features()
    unique_genres = sorted(list(set(genres)))
    label_map = {g:i for i,g in enumerate(unique_genres)}
    labels_int = np.array([label_map[g] for g in genres])
    
    train_loader, val_loader, n_classes = get_hard_dataloaders(audio, text, labels_int, hard_configs.batch_size)
    metrics_log = []

    # --- 2. Baselines ---
    print("\n>>> 2. Baselines...")
    flat_data = np.hstack([audio.reshape(len(audio), -1), text])
    flat_data = StandardScaler().fit_transform(flat_data)
    
    # PCA + KMeans
    pca_feat = PCA(n_components=32, random_state=SEED).fit_transform(flat_data)
    eval_pca = evaluate_clustering(pca_feat, labels_int, "Baseline (PCA+KMeans)")
    save_latent_plot(pca_feat, eval_pca.pop('labels_pred'), "t-SNE of PCA features + KMeans", "tsne_pca_baseline.png")
    metrics_log.append(eval_pca)
    
    # Spectral
    try:
        print("   Running Spectral Clustering...")
        spec = SpectralClustering(n_clusters=n_classes, affinity='nearest_neighbors', random_state=SEED)
        labels_spec = spec.fit_predict(pca_feat) 
        metrics_log.append({
            "Method": "Baseline (Spectral)",
            "Silhouette": silhouette_score(pca_feat, labels_spec),
            "NMI": normalized_mutual_info_score(labels_int, labels_spec),
            "ARI": adjusted_rand_score(labels_int, labels_spec),
            "Purity": cluster_purity(labels_int, labels_spec)
        })
        save_latent_plot(pca_feat, labels_spec, "t-SNE of PCA features + Spectral", "tsne_spectral_baseline.png")
    except Exception as e:
        print(f"   Spectral Failed: {e}")

    # Autoencoder
    print("   Training Autoencoder...")
    ae = SimpleAutoencoder(latent_dim=32)
    ae, _ = train_hard_task(ae, train_loader, val_loader, 40, 0.0, DEVICE, "Autoencoder", BASE_DIR)
    
    ae.eval()
    ae_feats = []
    with torch.no_grad():
        for i in range(0, len(audio), 32):
            ba = torch.tensor(audio[i:i+32]).to(DEVICE)
            bt = torch.tensor(text[i:i+32]).to(DEVICE)
            _, _, z = ae(ba, bt)
            ae_feats.append(z.cpu().numpy())
    ae_feats = np.vstack(ae_feats)
    eval_ae = evaluate_clustering(ae_feats, labels_int, "Baseline (Autoencoder)")
    save_latent_plot(ae_feats, eval_ae.pop('labels_pred'), "t-SNE of AE Latent Space + KMeans", "tsne_ae.png")
    metrics_log.append(eval_ae)
    save_reconstruction_plot(ae, val_loader, DEVICE, "Autoencoder")

    # --- 3. Beta-VAE ---
    print(f"\n>>> 3. Training Beta-VAE (Beta={hard_configs.beta_value})...")
    beta_vae = HybridVAE(latent_dim=32)
    beta_vae, _ = train_hard_task(beta_vae, train_loader, val_loader, hard_configs.epochs, hard_configs.beta_value, DEVICE, "BetaVAE", BASE_DIR)
    
    beta_vae.eval()
    beta_feats = []
    with torch.no_grad():
        for i in range(0, len(audio), 32):
            ba = torch.tensor(audio[i:i+32]).to(DEVICE)
            bt = torch.tensor(text[i:i+32]).to(DEVICE)
            _, _, mu, _ = beta_vae(ba, bt)
            beta_feats.append(mu.cpu().numpy())
    beta_feats = np.vstack(beta_feats)
    eval_beta = evaluate_clustering(beta_feats, labels_int, f"Beta-VAE (b={hard_configs.beta_value})")
    save_latent_plot(beta_feats, eval_beta.pop('labels_pred'), f"t-SNE of Beta-VAE (b={hard_configs.beta_value}) + KMeans", "tsne_beta.png")
    metrics_log.append(eval_beta)
    save_reconstruction_plot(beta_vae, val_loader, DEVICE, "BetaVAE")

    # --- 4. CVAE ---
    print("\n>>> 4. Training CVAE...")
    cvae = ConditionalVAE(n_classes=n_classes, latent_dim=32)
    cvae, _ = train_hard_task(cvae, train_loader, val_loader, hard_configs.epochs, 1.0, DEVICE, "CVAE", BASE_DIR)
    
    cvae.eval()
    cvae_feats = []
    with torch.no_grad():
        for i in range(0, len(audio), 32):
            end = min(i+32, len(audio))
            ba = torch.tensor(audio[i:end]).to(DEVICE)
            bt = torch.tensor(text[i:end]).to(DEVICE)
            
            batch_lbls = labels_int[i:end]
            onehot = np.zeros((len(batch_lbls), n_classes))
            onehot[np.arange(len(batch_lbls)), batch_lbls] = 1
            bl = torch.tensor(onehot, dtype=torch.float32).to(DEVICE)
            
            _, _, mu, _, _ = cvae(ba, bt, bl)
            cvae_feats.append(mu.cpu().numpy())
    cvae_feats = np.vstack(cvae_feats)
    
    eval_cvae = evaluate_clustering(cvae_feats, labels_int, "CVAE (Latent)")
    save_latent_plot(cvae_feats, eval_cvae.pop('labels_pred'), "t-SNE of CVAE Latent Space + KMeans", "tsne_cvae.png")
    metrics_log.append(eval_cvae)
    save_reconstruction_plot(cvae, val_loader, DEVICE, "CVAE", is_cvae=True)

    # --- Save ---
    df = pd.DataFrame(metrics_log)
    df.to_csv(os.path.join(BASE_DIR, "hard_task_metrics.csv"), index=False)
    print("\n>>> Hard Task Completed Successfully!")
    print(df)

if __name__ == "__main__":
    run_hard_task()