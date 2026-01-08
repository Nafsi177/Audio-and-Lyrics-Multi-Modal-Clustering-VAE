import torch
from src.easy_task.vae import BasicVAE
from src.easy_task.data_pipeline import get_dataloaders
from src.easy_task.preprocess import get_processed_features
from src.easy_task.feature_engineering import get_features
from src.easy_task.loss import ELBO_Loss
from tqdm import tqdm
import os
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_step_vae(model, data_loader, loss_fn, optimizer, device):
    model.train()
    train_loss = train_recon = train_kld = 0.0
    n = 0

    bar = tqdm(data_loader, desc="Training", leave=False)
    for xb in bar:
        xb = xb.to(device)

        optimizer.zero_grad()
        xhat, mu, logvar = model(xb)

        loss, recon, kld = loss_fn(xb, xhat, mu, logvar)

        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        train_loss += loss.item() * bs
        train_recon += recon.item() * bs
        train_kld += kld.item() * bs
        n += bs

        bar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon.item():.4f}",
            kld=f"{kld.item():.4f}",
        )

    return train_loss / n, train_recon / n, train_kld / n

def test_step_vae(model, data_loader, loss_fn, device):
    model.eval()
    val_loss = val_recon = val_kld = 0.0
    n = 0

    with torch.inference_mode():
        bar = tqdm(data_loader, desc="Validation", leave=False)
        for xb in bar:
            xb = xb.to(device)
            xhat, mu, logvar = model(xb)

            loss, recon, kld = loss_fn(xb, xhat, mu, logvar)

            bs = xb.size(0)
            val_loss += loss.item() * bs
            val_recon += recon.item() * bs
            val_kld += kld.item() * bs
            n += bs

            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kld=f"{kld.item():.4f}",
            )

    return val_loss / n, val_recon / n, val_kld / n


def train_vae(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs,
    device,
    checkpoint_gap=5,
):
    best_val_loss = float("inf")

    results = {
        "train_loss": [],
        "train_recon": [],
        "train_kld": [],
        "val_loss": [],
        "val_recon": [],
        "val_kld": [],
    }

    for epoch in range(1, epochs + 1):
        # beta = min(1.0, epoch / 20)  # KL warm-up
        # loss_fn.set_beta(beta)

        tr_loss, tr_rec, tr_kld = train_step_vae(
            model, train_loader, loss_fn, optimizer, device
        )

        va_loss, va_rec, va_kld = test_step_vae(
            model, val_loader, loss_fn, device
        )

        results["train_loss"].append(tr_loss)
        results["train_recon"].append(tr_rec)
        results["train_kld"].append(tr_kld)
        results["val_loss"].append(va_loss)
        results["val_recon"].append(va_rec)
        results["val_kld"].append(va_kld)

        print(
            f"Epoch {epoch:03d} | "
            f"Train {tr_loss:.4f} (R {tr_rec:.4f}, K {tr_kld:.4f}) | "
            f"Val {va_loss:.4f} (R {va_rec:.4f}, K {va_kld:.4f}) | "
            # f"β={beta:.2f}"
        )

        os.makedirs('./checkpoints/easy',exist_ok=True)
    
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), "checkpoints/easy/basic_vae_best.pt")

        if epoch % checkpoint_gap == 0:
            torch.save(model.state_dict(), f"checkpoints/easy/basic_vae_epoch_{epoch}.pt")

    return results,model


def initiate_basic_vae_training(epochs=100,checkpoint_gap=100,train_loader=None,val_loader=None):

    device="cuda" if torch.cuda.is_available() else "cpu"
    xb = next(iter(train_loader))   
    basic_vae=BasicVAE(input_dim=xb.shape[1],hidden_dim=128,latent_dim=8)
    optimizer=torch.optim.Adam(basic_vae.parameters(),lr=1e-3)
    loss_fn=ELBO_Loss()

    results,basic_vae = train_vae(
        model=basic_vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        checkpoint_gap=checkpoint_gap,
    )
    return results,basic_vae

if __name__=="__main__":
    features,genres,paths=get_features()
    scaled_features=get_processed_features(features)
    train_loader,val_loader=get_dataloaders(scaled_features)
    results,model=initiate_basic_vae_training(epochs=1000,checkpoint_gap=1000,train_loader=train_loader,val_loader=val_loader)