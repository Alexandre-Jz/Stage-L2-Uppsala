import os
import numpy as np
import torch
import optuna
from torch.utils.data import TensorDataset, random_split, DataLoader

from vilar_diffusion_CNN import MLPDiffusionModel, train_diffusion_model

# ===== Configuration statique =====
SIMULATION_BUDGETS      = [10000]
RUNS_PER_BUDGET         = 1
TRAIN_SPLIT             = 0.9
NUM_POSTERIOR_SAMPLES   = 10000
SEED                    = 42

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def run_hyperopt_for_budget(budget: int):
    # --- Chargement et découpe de données ---
    path = f'datasets/vilar_dataset_{budget}.npz'
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        return

    data = np.load(path, allow_pickle=True)
    theta_norm     = data['theta_norm'].astype(np.float32)
    ts_embed       = data['ts_embeddings'].astype(np.float32)
    true_theta     = data['true_theta']
    true_ts        = data['true_ts']
    theta_scaler   = data['theta_scaler'].item()
    true_ts_embed  = data['true_ts_embedding'].astype(np.float32)

    θ = torch.from_numpy(theta_norm)
    y = torch.from_numpy(ts_embed)
    full_ds = TensorDataset(θ, y)
    n_train = int(TRAIN_SPLIT * len(full_ds))
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(SEED)
    )

    theta_dim = theta_norm.shape[1]

    # --- Fonction objectif pour Optuna ---
    def objective(trial: optuna.Trial) -> float:
        # 1) Échantillonnage des hyperparamètres
        hidden_dim    = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        num_layers    = trial.suggest_int("num_layers", 2, 8)
        num_timesteps = trial.suggest_categorical("num_timesteps", [100, 200, 300])
        batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        patience      = trial.suggest_int("patience", 5, 50)
        lr_patience   = trial.suggest_int("lr_patience", 1, 20)
        lr_factor     = trial.suggest_float("lr_factor", 0.1, 0.9)
        min_lr        = trial.suggest_float("min_lr", 1e-7, 1e-3, log=True)
        num_epochs    = 1000  # on fixe le maximum

        # 2) DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        # 3) Modèle
        model = MLPDiffusionModel(
            theta_dim=theta_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps
        ).to(device)

        # 4) Entraînement
        best_model, train_losses, val_losses = train_diffusion_model(
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

        val_min = float(min(val_losses))
        print(f"  Trial {trial.number:2d} → val_loss {val_min:.4e}")
        return val_min

    # --- Lancement d'Optuna avec barre de progression ---
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=20)

    # --- Résultats ---
    print("\n>> Best trial parameters:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    print(f"--> Best validation loss: {study.best_value:.4e}\n")

    # --- Entraînement final et sampling ---
    bp = study.best_params
    train_loader = DataLoader(train_ds, batch_size=bp["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bp["batch_size"])
    final_model = MLPDiffusionModel(
        theta_dim=theta_dim,
        hidden_dim=bp["hidden_dim"],
        num_layers=bp["num_layers"],
        num_timesteps=bp["num_timesteps"]
    ).to(device)

    best_model, _, _ = train_diffusion_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,
        patience=bp["patience"],
        learning_rate=bp["learning_rate"],
        weight_decay=bp["weight_decay"],
        lr_patience=bp["lr_patience"],
        lr_factor=bp["lr_factor"],
        min_lr=bp["min_lr"]
    )

    best_model.eval()
    y_obs = torch.from_numpy(true_ts_embed).to(device)
    y_obs = y_obs.repeat(NUM_POSTERIOR_SAMPLES, 1)
    with torch.no_grad():
        post = best_model.sample(NUM_POSTERIOR_SAMPLES, y_obs, temperature=0.8).cpu().numpy()
    post = theta_scaler.inverse_transform(post)

    out_path = f"posterior_samples/vilar_optuna_{budget}.npz"
    np.savez_compressed(
        out_path,
        posterior_samples=post,
        true_theta=true_theta,
        true_ts=true_ts,
        best_params=study.best_params,
        best_val_loss=study.best_value
    )
    print(f"Saved optimized posterior samples to {out_path}\n")


if __name__ == "__main__":
    for budget in SIMULATION_BUDGETS:
        run_hyperopt_for_budget(budget)
