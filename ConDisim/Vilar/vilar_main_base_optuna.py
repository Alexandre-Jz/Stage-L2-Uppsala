import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from vilar_diffusion_base import MLPDiffusionModel, create_dataloaders

# ===== Configuration =====
SIMULATION_BUDGETS = [10000]  # Simulation budgets to consider
TRAIN_SPLIT = 0.9             # Fraction of data used for training

# Default training settings
NUM_EPOCHS = 1000             # Maximum number of training epochs
LR_FACTOR = 0.5               # Learning rate reduction factor
LR_PATIENCE = 5               # Learning rate scheduler patience
MIN_LR = 1e-5                 # Minimum learning rate
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure output directories
os.makedirs('vilar_optuna_plots', exist_ok=True)
os.makedirs('optuna_study', exist_ok=True)

# Objective function for Optuna
def objective(trial):
    # Hyperparameter sampling
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=64)
    num_timesteps = trial.suggest_int('num_timesteps', 100, 500, step=100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    patience = trial.suggest_int('patience', 5, 50, step=5)

    # Use the first budget for hyperparameter tuning
    budget = SIMULATION_BUDGETS[0]
    dataset_path = f'datasets/vilar_dataset_{budget}.npz'
    data = np.load(dataset_path, allow_pickle=True)
    theta_norm = data['theta_norm']
    ts_embeddings = data['ts_embeddings']

    # Prepare dataloaders
    train_loader, val_loader = create_dataloaders(
        theta=theta_norm,
        y=ts_embeddings,
        batch_size=batch_size,
        train_split=TRAIN_SPLIT
    )

    # Initialize model
    model = MLPDiffusionModel(
        theta_dim=theta_norm.shape[1],
        y_dim=ts_embeddings.shape[1],
        hidden_dim=hidden_dim,
        num_timesteps=num_timesteps
    ).to(DEVICE)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=False
    )
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    early_stop_counter = 0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            # Forward diffusion loss
            t = torch.randint(0, model.num_timesteps, (theta_batch.size(0),), device=DEVICE)
            noise = torch.randn_like(theta_batch)
            alpha_hat_t = model.alpha_hat[t].unsqueeze(-1)
            theta_noisy = torch.sqrt(alpha_hat_t) * theta_batch + torch.sqrt(1 - alpha_hat_t) * noise
            noise_pred = model(theta_noisy, y_batch, t)
            loss = mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                t = torch.randint(0, model.num_timesteps, (theta_batch.size(0),), device=DEVICE)
                noise = torch.randn_like(theta_batch)
                alpha_hat_t = model.alpha_hat[t].unsqueeze(-1)
                theta_noisy = torch.sqrt(alpha_hat_t) * theta_batch + torch.sqrt(1 - alpha_hat_t) * noise
                noise_pred = model(theta_noisy, y_batch, t)
                val_loss += mse_loss(noise_pred, noise).item()

        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

    return best_val_loss


def main():
    # Display device info
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    # Save and print best trial
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, val in trial.params.items():
        print(f"    {key}: {val}")

    # Save study for later analysis
    optuna.study.trial.serialize(study, 'optuna_study/vilar_optuna_study.pkl')

if __name__ == '__main__':
    main()
