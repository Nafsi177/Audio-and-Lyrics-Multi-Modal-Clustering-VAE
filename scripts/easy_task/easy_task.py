from src.easy_task.training_pipeline import initiate_basic_vae_training
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from src.easy_task.data_pipeline import get_dataloaders
from src.easy_task.preprocess import get_processed_features
from src.easy_task.feature_engineering import get_features
import torch
import numpy as np
import pandas as pd
import os


os.makedirs('./results/easy_task/latent_visualization', exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#  Get Features 
features, genres, paths = get_features()
scaled_features = get_processed_features(features)
train_loader, val_loader = get_dataloaders(scaled_features)

print("For Now we are using audio and mfcc")
K = 5
SEED = 42

# PCA + KMeans baseline 
pca = PCA(n_components=10, random_state=SEED)
Xp = pca.fit_transform(scaled_features)

kmeans_pca = KMeans(n_clusters=K, n_init="auto", random_state=SEED)
labels_pca = kmeans_pca.fit_predict(Xp)

sil_pca = silhouette_score(Xp, labels_pca)
ch_pca = calinski_harabasz_score(Xp, labels_pca)

# t-SNE on PCA features (cluster-colored) 
tsne_pca = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED, learning_rate="auto")
Xp_2d = tsne_pca.fit_transform(Xp)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    Xp_2d[:,0], Xp_2d[:,1], 
    c=labels_pca, 
    cmap=plt.get_cmap("rainbow"), 
    s=30, 
    edgecolor='k', 
    alpha=0.8
)
plt.title("t-SNE of PCA features + KMeans")
plt.colorbar(scatter, ticks=range(len(set(labels_pca))), label="Cluster")
plt.savefig('./results/easy_task/latent_visualization/pca_kmeans(tsne).png')
plt.close()

# Train VAE 
results, basic_vae = initiate_basic_vae_training(
    epochs=1000, checkpoint_gap=1000,
    train_loader=train_loader, val_loader=val_loader
)

# Plot training curve 
train_hist = np.array(results['train_loss'])
val_hist = np.array(results['val_loss'])

plt.figure(figsize=(7,4))
plt.plot(train_hist[:], label="train loss")
plt.plot(val_hist[:], label="val loss")
plt.title("VAE Training Loss")
plt.legend()
plt.savefig('./results/easy_task/vae_traintime_metric.png')
plt.close()

# Extract latent space 
basic_vae.eval()
with torch.no_grad():
    X_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
    h = basic_vae.enc(X_tensor)
    Z_mu = basic_vae.mu(h).cpu().numpy()

print("Latent shape:", Z_mu.shape)

# KMeans on latent space 
kmeans_latent = KMeans(n_clusters=K, n_init=10, random_state=SEED)
labels_latent = kmeans_latent.fit_predict(Z_mu)

sil_latent = silhouette_score(Z_mu, labels_latent)
ch_latent = calinski_harabasz_score(Z_mu, labels_latent)

print("VAE(latent mu)+KMeans")
print(" Silhouette:", round(sil_latent, 4))
print(" Calinski-Harabasz:", round(ch_latent, 2))

# Save metrics 
metrics = pd.DataFrame([
    {"method": "PCA(10)+KMeans", "no of cluster(K)": K, "silhouette": sil_pca, "calinski_harabasz": ch_pca},
    {"method": "VAE(latent_mu)+KMeans", "no of cluster(K)": K, "silhouette": sil_latent, "calinski_harabasz": ch_latent},
])
metrics.to_csv('./results/easy_task/clustering_metric.csv', index=False)
print(metrics)

# t-SNE on latent space (cluster-colored) 
tsne_latent = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED, learning_rate="auto")
Z_2d = tsne_latent.fit_transform(Z_mu)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    Z_2d[:,0], Z_2d[:,1],
    c=labels_latent, 
    cmap=plt.get_cmap("rainbow"),
    s=30,
    edgecolor='k',
    alpha=0.8
)
plt.title("t-SNE of VAE Latent Space + KMeans")
plt.colorbar(scatter, ticks=range(len(set(labels_latent))), label="Cluster")
plt.savefig('./results/easy_task/latent_visualization/VAE_kmeans(tsne).png')
plt.close()









#  t-SNE on latent space (genre-colored) 
# Map genres to integers for coloring
# genres_array = np.array(genres)
# unique_genres = np.unique(genres_array)
# genre_to_int = {g: i for i, g in enumerate(unique_genres)}
# genre_labels = np.array([genre_to_int[g] for g in genres_array])


# plt.figure(figsize=(8,6))
# scatter = plt.scatter(
#     Z_2d[:,0], Z_2d[:,1],
#     c=genre_labels,
#     cmap=plt.get_cmap("tab20"),
#     s=30,
#     edgecolor='k',
#     alpha=0.8
# )
# plt.title("t-SNE of VAE Latent Space (colored by true genre)")
# cbar = plt.colorbar(scatter, ticks=range(len(unique_genres)))
# cbar.ax.set_yticklabels(unique_genres)
# cbar.set_label("True Genre")
# plt.savefig('./results/easy_task/latent_visualization/VAE_by_genre(tsne).png')
# plt.close()
