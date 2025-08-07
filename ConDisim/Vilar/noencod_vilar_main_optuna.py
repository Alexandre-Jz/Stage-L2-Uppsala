import os
import datetime
import numpy as np
import torch
import optuna
from torch.utils.data import TensorDataset, DataLoader

from noencod_vilar_diffusion import (
    MLPFiLMDiffusion as DiffusionModel,
    train_diffusion_model,
    create_dataloaders,
)

# ===== Fixed Config =====
SIMULATION_BUDGETS    = [10000]
RUNS_PER_BUDGET       = 1
TRAIN_SPLIT           = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data directories
DATASET_DIR      = "datasets"
MODEL_DIR        = "models"
POST_SAMPLES_DIR = "noencod_posterior_samples"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(POST_SAMPLES_DIR, exist_ok=True)

# Optuna settings
N_TRIALS = 20
SEED     = 42

# Posterior sampling
NUM_POSTERIOR_SAMPLES = 1000
SAMPLING_TEMPERATURE  = 0.9


def load_dataset(budget):
    path = os.path.join(DATASET_DIR, f"vilar_dataset_{budget}_noencod.npz")
    data = np.load(path, allow_pickle=True)
    theta      = data["theta"].astype(np.float32)
    ts_data    = data["ts_data"].astype(np.float32)
    true_ts    = data["true_ts"].astype(np.float32)
    # squeeze optional channel dim
    if ts_data.ndim == 4 and ts_data.shape[1] == 1:
        ts_data = ts_data.squeeze(1)
    # normalize theta
    theta_mean = theta.mean(axis=0, keepdims=True)
    theta_std  = theta.std(axis=0, keepdims=True) + 1e-6
    theta_norm = (theta - theta_mean) / theta_std
    return theta, theta_norm, theta_mean, theta_std, ts_data, true_ts


def objective(trial, theta_norm, ts_embeddings):
    # Hyperparameter search space
    hidden_dim    = trial.suggest_categorical('hidden_dim',    [128, 256, 512, 768])
    num_layers    = trial.suggest_int('num_layers',    2, 8)
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 200, 500])
    batch_size    = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_epochs    = trial.suggest_int('num_epochs', 100, 1000)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay  = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    patience      = trial.suggest_int('patience', 5, 50)
    lr_patience   = trial.suggest_int('lr_patience', 1, 10)
    lr_factor     = trial.suggest_float('lr_factor', 0.1, 0.9)
    min_lr        = trial.suggest_loguniform('min_lr', 1e-6, 1e-3)

    # Create dataloaders with train/val split
    train_loader, val_loader = create_dataloaders(
        theta=theta_norm,
        y=ts_embeddings,
        batch_size=batch_size,
        train_split=TRAIN_SPLIT
    )

    # Instantiate model with trial hyperparams
    model = DiffusionModel(
        theta_dim=theta_norm.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
    ).to(DEVICE)

    # Train
    _, _, val_losses = train_diffusion_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        min_lr=min_lr
    )

    # Return final validation loss
    return val_losses[-1]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for budget in SIMULATION_BUDGETS:
        print(f"\n=== Budget: {budget} ===")
        # Load dataset
        theta, theta_norm, theta_mean, theta_std, ts_embeddings, true_ts = load_dataset(budget)

        # Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, theta_norm, ts_embeddings), n_trials=N_TRIALS)
        best = study.best_params
        print("Best hyperparameters:", best)

        # Final training with best hyperparams
        model = DiffusionModel(
            theta_dim=theta_norm.shape[1],
            hidden_dim=best['hidden_dim'],
            num_layers=best['num_layers'],
            num_timesteps=best['num_timesteps'],
        ).to(DEVICE)
        train_loader, val_loader = create_dataloaders(
            theta=theta_norm,
            y=ts_embeddings,
            batch_size=best['batch_size'],
            train_split=TRAIN_SPLIT
        )
        best_model, _, _ = train_diffusion_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=best['num_epochs'],
            patience=best['patience'],
            learning_rate=best['learning_rate'],
            weight_decay=best['weight_decay'],
            lr_patience=best['lr_patience'],
            lr_factor=best['lr_factor'],
            min_lr=best['min_lr']
        )

        # Posterior sampling with pre-extracted features
        best_model.eval()
        y_raw = torch.from_numpy(true_ts).float().unsqueeze(0).to(DEVICE)
        y_feat = best_model.feature_extractor(y_raw).repeat(NUM_POSTERIOR_SAMPLES, 1)
        with torch.no_grad():
            posterior_samples = best_model.sample(
                NUM_POSTERIOR_SAMPLES, y_feat, temperature=SAMPLING_TEMPERATURE
            ).cpu().numpy()
        posterior_samples = posterior_samples * theta_std + theta_mean

        # Save
        ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(POST_SAMPLES_DIR, f'vilar_post_{budget}_{ts_str}.npz')
        np.savez_compressed(
            out_path,
            posterior_samples=posterior_samples,
            true_theta=theta,
            true_ts=true_ts,
            best_params=best
        )
        print(f"Saved posterior to {out_path}")

if __name__ == '__main__':
    main()
