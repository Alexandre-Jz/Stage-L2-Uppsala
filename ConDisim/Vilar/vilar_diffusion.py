import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def noise_schedule(num_timesteps, beta_start=0.00001, beta_end=0.005, schedule='quadratic'):
    if schedule == 'linear':
        beta = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    elif schedule == 'quadratic':
        beta = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
    elif schedule == 'cosine':
        beta = torch.tensor(
            [(1 - math.cos((t / num_timesteps) * math.pi/2))**2 for t in range(num_timesteps)],
            dtype=torch.float32
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")
    
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    return beta, alpha, alpha_hat

class MLPDiffusionModel(nn.Module):
    def __init__(self, theta_dim, y_dim, hidden_dim=256, num_timesteps=500):
        super(MLPDiffusionModel, self).__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.num_timesteps = num_timesteps
        
        # Input dimension: theta + y + timestep
        in_dim = theta_dim + y_dim + 1
        
        # Simple MLP with residual connections
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.final = nn.Linear(hidden_dim, theta_dim)
        
        # Initialize noise schedule
        beta, alpha, alpha_hat = noise_schedule(num_timesteps)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_hat', alpha_hat)

    def forward(self, theta, y, t):
        # Scale timestep to [0,1]
        t = t.float().unsqueeze(-1) / float(self.num_timesteps)
        x = torch.cat([theta, y, t], dim=1)
        
        # Forward pass with residual connections
        h1 = self.layer1(x)
        h2 = self.layer2(h1) + h1
        h3 = self.layer3(h2) + h2
        h4 = self.layer4(h3) + h3
        return self.final(h4)

    def sample(self, N, y_observed, temperature=0.8):
        with torch.no_grad():
            # Initial noise from standard normal (matching forward process)
            theta_samples = torch.randn((N, self.theta_dim), device=device) * temperature
            
            for t in reversed(range(self.num_timesteps-1, -1, -1)):
                t_tensor = torch.full((N,), t, device=device, dtype=torch.long)
                
                alpha_hat_t = self.alpha_hat[t].unsqueeze(-1)
                alpha_t = self.alpha[t].unsqueeze(-1)
                beta_t = self.beta[t].unsqueeze(-1)
                
                # Predict noise
                noise_pred = self.forward(theta_samples, y_observed, t_tensor)
                
                # Mean calculation matching forward process
                mu_t = (theta_samples - (beta_t / torch.sqrt(1 - alpha_hat_t)) * noise_pred) / torch.sqrt(alpha_t)
                
                if t > 0:  # Only add noise if not the final step
                    z = torch.randn_like(theta_samples) * temperature
                    theta_samples = mu_t + torch.sqrt(beta_t) * z
                else:
                    theta_samples = mu_t
                    
            return theta_samples

def diffusion_loss(model, theta_0, y, num_timesteps, mse_loss):
    batch_size = theta_0.size(0)
    t = torch.randint(0, num_timesteps, (batch_size,), device=device)
    noise = torch.randn_like(theta_0)
    
    alpha_hat_t = model.alpha_hat[t].unsqueeze(-1)
    theta_noisy = torch.sqrt(alpha_hat_t) * theta_0 + torch.sqrt(1 - alpha_hat_t) * noise
    
    noise_pred = model(theta_noisy, y, t)
    return mse_loss(noise_pred, noise)

def train_diffusion_model(model, train_loader, val_loader, num_epochs=1000, patience=20):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        cooldown=5,  # Increased cooldown period
        min_lr=1e-7,
        verbose=True  # Added verbosity for better monitoring
    )
    
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            loss = diffusion_loss(model, theta_batch, y_batch, model.num_timesteps, mse_loss)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)
                loss = diffusion_loss(model, theta_batch, y_batch, model.num_timesteps, mse_loss)
                epoch_val_loss += loss.item()
                
        val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                model = best_model
                break
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Training Progress')
    plt.legend()
    plt.savefig('diffusion_training.pdf')
    plt.close()
    
    return best_model, train_losses, val_losses

def create_dataloaders(theta, y, batch_size=64, train_split=0.9):
    """Create train and validation dataloaders."""
    # Convert to tensors
    theta_tensor = torch.FloatTensor(theta)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset
    dataset = TensorDataset(theta_tensor, y_tensor)
    
    # Split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
