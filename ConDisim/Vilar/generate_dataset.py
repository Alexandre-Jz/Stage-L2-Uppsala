import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from vilar_main import simulator, generate_data_parallel, train_autoencoder

def main():
    # Clear terminal
    os.system('clear')
    
    # Create dataset directory if it doesn't exist
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Step 1: Generate true/observed data
    print("\nGenerating true/observed data...")
    true_theta = np.array([50, 500, 0.01, 50, 50, 5, 10, 0.5, 1, 0.2, 1, 1, 2, 50, 100])  # Default parameters
    true_ts = simulator(true_theta)  # Generate single trajectory with true parameters
    true_ts = true_ts.squeeze(0)  # Remove the extra dimension (1,3,200) -> (3,200)
    true_ts = true_ts[np.newaxis, :]  # Add batch dimension back (3,200) -> (1,3,200)
    print(f"True data shapes - Time series: {true_ts.shape}, Parameters: {true_theta.shape}")
    
    # Step 2: Generate training data for different budgets
    budgets = [100, 500, 1000]  # Three different budgets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for budget in budgets:
        print(f"\nProcessing budget: {budget}")
        
        # Generate training data and parameters
        ts_data, theta = generate_data_parallel(budget)
        ts_data = ts_data.squeeze(1)  # Remove the extra dimension (N,1,3,200) -> (N,3,200)
        print(f"Training data shapes - Time series: {ts_data.shape}, Parameters: {theta.shape}")
        
        # Create and fit separate scalers
        data_scaler = MinMaxScaler()
        theta_scaler = MinMaxScaler()
        
        # Normalize training time series data
        ts_data_flat = ts_data.reshape(ts_data.shape[0], -1)
        ts_data_norm = data_scaler.fit_transform(ts_data_flat)
        ts_data_norm = ts_data_norm.reshape(ts_data.shape)
        
        # Normalize training theta values
        theta_norm = theta_scaler.fit_transform(theta)
        
        # Train autoencoder and get summary statistics (latent representations)
        model, summary_stats = train_autoencoder(ts_data_norm, device=device)
        
        # Get summary statistics for true data
        true_ts_flat = true_ts.reshape(true_ts.shape[0], -1)
        true_ts_norm = data_scaler.transform(true_ts_flat)
        true_ts_norm = true_ts_norm.reshape(true_ts.shape)
        
        # Convert to tensor and get latent representation
        true_ts_tensor = torch.FloatTensor(true_ts_norm).to(device)
        with torch.no_grad():
            true_summary_stats = model.encode(true_ts_tensor).cpu().numpy()
        
        # Save the dataset
        save_file = os.path.join(dataset_dir, f'vilar_dataset_budget_{budget}.npz')
        np.savez(save_file, 
                 observed_data=true_ts,          # True/observed time series
                 true_parameters=true_theta,      # True parameters
                 observed_stats=true_summary_stats,  # Summary stats of true data
                 training_data=ts_data,          # Training time series
                 training_parameters=theta,       # Training parameters
                 training_stats=summary_stats,    # Training summary stats
                 data_scaler=data_scaler,
                 theta_scaler=theta_scaler)
        
        print(f"Completed processing for budget {budget}")
        print(f"Data shapes:")
        print(f"  True/Observed - Data: {true_ts.shape}, Parameters: {true_theta.shape}, Stats: {true_summary_stats.shape}")
        print(f"  Training     - Data: {ts_data.shape}, Parameters: {theta.shape}, Stats: {summary_stats.shape}")
        print(f"Saved dataset to {save_file}")

if __name__ == '__main__':
    main()
