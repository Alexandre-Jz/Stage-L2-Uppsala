import torch
import numpy as np
import math
import sbibm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

# Load data based on task name
def load_data(task_name, num_samples):
    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()
    x_o = task.get_observation(num_observation=1)

    thetas = prior(num_samples=num_samples)  # Sample theta parameters from the prior
    y_data = simulator(thetas)  # Generate data using the simulator

    # Scale the data
    theta_scaler = StandardScaler()
    y_scaler = StandardScaler()
    thetas = theta_scaler.fit_transform(thetas)
    y_data = y_scaler.fit_transform(y_data)

    # Convert data to tensors
    thetas = torch.tensor(thetas, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)

    return thetas, y_data, theta_scaler, y_scaler, task

# Create dataset and DataLoader
def create_dataloaders(thetas, y_data, batch_size):
    dataset = TensorDataset(thetas, y_data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Noise schedule with options
def noise_schedule(num_timesteps, beta_start, beta_end, beta_schedule="linear"):
    if beta_schedule == "linear":
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    elif beta_schedule == "quadratic":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(num_timesteps, lambda t: torch.cos(t.clone().detach().float() + 0.008 / 1.008 * math.pi / 2) ** 2).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError("Invalid beta_schedule. Choose 'linear', 'quadratic', or 'cosine'.")

# Function for cosine beta schedule
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        t1_tensor = torch.tensor(t1, dtype=torch.float32)
        t2_tensor = torch.tensor(t2, dtype=torch.float32)
        betas.append(min(1 - alpha_bar(t2_tensor) / alpha_bar(t1_tensor), max_beta))
    return torch.tensor(betas, dtype=torch.float32)
