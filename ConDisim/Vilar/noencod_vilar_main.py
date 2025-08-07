import os
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from noencod_vilar_diffusion import (
    MLPFiLMDiffusion as DiffusionModel, train_diffusion_model,
    create_dataloaders,
)

# ============ Configurable Parameters ============
# Data generation and splits
SIMULATION_BUDGETS     = [20000,30000]   # Number of simulations per budget
RUNS_PER_BUDGET        = 1         # Number of independent runs per budget
TRAIN_SPLIT            = 0.9       # Fraction of data used for training

# Model architecture
HIDDEN_DIM             = 128      # Hidden dimension of MLP layers
NUM_LAYERS             = 6        # Number of MLP layers
NUM_TIMESTEPS          = 500      # Number of diffusion steps

# Training parameters
BATCH_SIZE             = 32       # Batch size for training
NUM_EPOCHS             = 1000     # Maximum number of training epochs
LEARNING_RATE          = 0.0028417072368355158     # Initial learning rate
WEIGHT_DECAY           = 1.39742884805759e-05     # Weight decay for AdamW
PATIENCE               = 27       # Early stopping patience
LR_PATIENCE            = 10        # LR scheduler patience
LR_FACTOR              = 0.3511869926187101      # LR reduction factor on plateau
MIN_LR                 = 0.00020507162259327777     # Minimum LR for scheduler

# Posterior sampling parameters
NUM_POSTERIOR_SAMPLES  = 1000     # Number of posterior samples
SAMPLING_TEMPERATURE   = 1      # Sampling temperature

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device → {DEVICE}")

# Directories
DATASET_DIR      = "datasets"
MODEL_DIR        = "models"
POST_SAMPLES_DIR = "noencod_posterior_samples"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(POST_SAMPLES_DIR, exist_ok=True)

    for budget in SIMULATION_BUDGETS:
        print("\n" + "="*60)
        print(f"Processing budget: {budget}")
        print("="*60)

        ds_path = os.path.join(DATASET_DIR, f"vilar_dataset_{budget}_noencod.npz")
        if not os.path.exists(ds_path):
            print(f"Dataset not found: {ds_path}")
            continue

        data    = np.load(ds_path, allow_pickle=True)
        theta   = data["theta"].astype(np.float32)      # (N, θ_dim)
        ts_data = data["ts_data"].astype(np.float32)    # (N, features, timesteps)
        true_ts = data["true_ts"].astype(np.float32)    # (features, timesteps)

        if ts_data.ndim == 4 and ts_data.shape[1] == 1:
            ts_data = ts_data.squeeze(1)

        theta_mean = theta.mean(axis=0, keepdims=True)
        theta_std  = theta.std(axis=0, keepdims=True) + 1e-6
        theta_norm = (theta - theta_mean) / theta_std

        train_loader, val_loader = create_dataloaders(
            theta=theta_norm,
            y=ts_data,
            batch_size=BATCH_SIZE,
            train_split=TRAIN_SPLIT,
        )

        for run in range(RUNS_PER_BUDGET):
            print(f"\nRun {run+1}/{RUNS_PER_BUDGET} for budget {budget}")

            model = DiffusionModel(
                theta_dim=theta.shape[1],
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                num_timesteps=NUM_TIMESTEPS,
            ).to(DEVICE)

            best_model, tr_losses, val_losses = train_diffusion_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=NUM_EPOCHS,
                patience=PATIENCE,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                lr_patience=LR_PATIENCE,
                lr_factor=LR_FACTOR,
                min_lr=MIN_LR,
            )

            if best_model is None:
                print("⚠ No model retained by early-stopping; skipping sampling.")
                continue

            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"vilar_model_{budget}_run{run+1}_{ts_str}.pt"
            model_path = os.path.join(MODEL_DIR, model_name)
            torch.save(best_model.state_dict(), model_path)
            print(f"Model saved → {model_path}")

            # --- Posterior sampling ---
            best_model.eval()
            # Pré-extraction des features pour true_ts
            y_raw = torch.from_numpy(true_ts).float().unsqueeze(0).to(DEVICE)      # (1, features, timesteps)
            y_feat = best_model.feature_extractor(y_raw)                          # (1, feature_dim)
            y_feat = y_feat.repeat(NUM_POSTERIOR_SAMPLES, 1)                       # (N_samples, feature_dim)

            with torch.no_grad():
                posterior = best_model.sample(
                    NUM_POSTERIOR_SAMPLES,
                    y_feat,
                    temperature=SAMPLING_TEMPERATURE
                ).cpu().numpy()

            posterior = posterior * theta_std + theta_mean

            out_name = f"vilar_post_{budget}_{ts_str}.npz"
            out_path = os.path.join(POST_SAMPLES_DIR, out_name)
            np.savez_compressed(
                out_path,
                posterior_samples=posterior,
                true_theta=data["true_theta"],
                budget=budget
            )
            print(f"Posterior samples saved → {out_path} (shape {posterior.shape})")


if __name__ == "__main__":
    main()