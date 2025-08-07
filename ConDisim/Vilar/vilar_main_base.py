import numpy as np
import torch
import os
from vilar_diffusion_base import MLPDiffusionModel, train_diffusion_model, create_dataloaders
import datetime

# ===== Configuration =====
# Data parameters
SIMULATION_BUDGETS = [10000,20000,30000]  # Number of simulations per budget
RUNS_PER_BUDGET = 1          # Number of independent runs per budget
TRAIN_SPLIT = 0.9           # Fraction of data used for training

# Model architecture
HIDDEN_DIM = 128 # Hidden dimension of MLP layers
NUM_TIMESTEPS = 500         # Number of diffusion steps

# Training parameters
BATCH_SIZE = 32             # Batch size for training
NUM_EPOCHS = 1000           # Maximum number of training epochs
LEARNING_RATE = 0.0035688626277862226        # Initial learning rate
WEIGHT_DECAY = 4.952067876224309e-06         # Weight decay for AdamW
PATIENCE = 25               # Early stopping patience
LR_PATIENCE = 5             # Learning rate scheduler patience
LR_FACTOR = 0.5            # Learning rate reduction factor
MIN_LR = 1e-5              # Minimum learning rate

# Sampling parameters
NUM_POSTERIOR_SAMPLES = 10000  # Number of posterior samples to generate

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

def main():
    # Create necessary directories
    os.makedirs('vilar_plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('posterior_samples', exist_ok=True)
    
    for budget in SIMULATION_BUDGETS:
        print(f"\n{'='*50}")
        print(f"Processing simulation budget: {budget}")
        print(f"{'='*50}")
        
        try:
            # Load dataset
            dataset_path = f'datasets/vilar_dataset_{budget}.npz'
            if not os.path.exists(dataset_path):
                print(f"Dataset not found: {dataset_path}")
                continue
                
            data = np.load(dataset_path, allow_pickle=True)  # Allow loading Python objects (scalers)
            theta_norm = data['theta_norm']  # Already normalized parameters
            ts_embeddings = data['ts_embeddings']  # Time series embeddings
            true_theta = data['true_theta']  # Original scale parameters
            true_ts = data['true_ts']  # Original time series
            true_ts_embedding = data['true_ts_embedding']  # True time series embedding
            theta_scaler = data['theta_scaler'].item()  # Get scaler
            
            print(f"Loaded data shapes:")
            print(f"- theta_norm: {theta_norm.shape}")
            print(f"- ts_embeddings: {ts_embeddings.shape}")
            print(f"- true_ts_embedding: {true_ts_embedding.shape}")
            
            # Create dataloaders with normalized parameters and embeddings
            train_loader, val_loader = create_dataloaders(
                theta=theta_norm,  # Normalized parameters
                y=ts_embeddings,  # Time series embeddings
                batch_size=BATCH_SIZE,
                train_split=TRAIN_SPLIT
            )
            
            for run in range(RUNS_PER_BUDGET):
                print(f"\nRun {run + 1}/{RUNS_PER_BUDGET}")
                
                try:
                    # Initialize model
                    model = MLPDiffusionModel(
                        theta_dim=theta_norm.shape[1],  # Parameter dimension
                        y_dim=ts_embeddings.shape[1],  # Embedding dimension
                        hidden_dim=HIDDEN_DIM,
                        num_timesteps=NUM_TIMESTEPS
                    ).to(device)
                    
                    # Train model with normalized data
                    best_model, train_losses, val_losses = train_diffusion_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        num_epochs=NUM_EPOCHS,
                        patience=PATIENCE
                    )
                    
                    # Generate samples using true time series embedding
                    best_model.eval()
                    with torch.no_grad():
                        # Use true time series embedding for conditioning
                        y_obs = torch.FloatTensor(true_ts_embedding).to(device)  # Shape: [1, embedding_dim]
                        y_obs = y_obs.repeat(NUM_POSTERIOR_SAMPLES, 1)  # Shape: [num_samples, embedding_dim]
                        posterior_samples = best_model.sample(NUM_POSTERIOR_SAMPLES, y_obs, temperature=0.8)
                        posterior_samples = posterior_samples.cpu().numpy()
                    
                    # Denormalize posterior samples using the scaler
                    posterior_samples = theta_scaler.inverse_transform(posterior_samples)
                    
                    # Print first 5 posterior samples and true parameters
                    print("\nFirst 5 posterior samples:")
                    print("-" * 50)
                    for i in range(5):
                        print(f"Sample {i+1}: {posterior_samples[i]}")
                    print("\nTrue parameters:")
                    print(f"{true_theta}")
                    
                    # Calculate and print mean and std of posterior samples
                    posterior_mean = np.mean(posterior_samples, axis=0)
                    posterior_std = np.std(posterior_samples, axis=0)
                    print("\nPosterior statistics:")
                    print("-" * 50)
                    print(f"Mean: {posterior_mean}")
                    print(f"Std:  {posterior_std}")
                    
                    # Save results
                    save_dict = {
                        'posterior_samples': posterior_samples,  # Denormalized
                        'true_theta': true_theta,  # Original scale
                        'true_ts': true_ts,  # Original scale
                        'true_ts_embedding': true_ts_embedding,  # Time series embedding
                        'budget': budget
                    }
                    
                    posterior_samples_path = f'posterior_samples/vilar_posterior_{budget}_run_{run + 1}.npz'
                    np.savez_compressed(posterior_samples_path, **save_dict)
                    print(f"\nSaved results to {posterior_samples_path}")
                    print(f"Posterior samples shape: {posterior_samples.shape}")
                    
                except Exception as e:
                    print(f"Error in run {run + 1}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing budget {budget}: {str(e)}")
            continue

if __name__ == '__main__':
    main()