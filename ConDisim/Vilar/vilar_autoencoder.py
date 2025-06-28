import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=15):
        super(Conv1DAutoencoder, self).__init__()
        
        # ------------------
        # Encoder
        # ------------------
        # Input shape: (B, 3, 200)
        # We'll downsample from length 200 -> 100 -> 50 -> 25
        self.encoder = nn.Sequential(
            # (200 -> 100)
            nn.Conv1d(in_channels=input_channels, out_channels=32,
                      kernel_size=4, stride=2, padding=1),   # => (B,32,100)
            nn.LeakyReLU(0.1),
            
            # (100 -> 50)
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2, padding=1),   # => (B,64,50)
            nn.LeakyReLU(0.1),
            
            # (50 -> 25)
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2, padding=1),   # => (B,128,25)
            nn.LeakyReLU(0.1)
        )
        
        # Now flatten: (B,128,25) -> (B,128*25) -> (B,latent_dim)
        self.enc_fc = nn.Linear(128 * 25, latent_dim)
        
        # ------------------
        # Decoder
        # ------------------
        # We'll do the reverse: (B,latent_dim) -> (B,128*25) -> (B,128,25)
        self.dec_fc = nn.Linear(latent_dim, 128 * 25)
        
        # Now we upsample from length 25 -> 50 -> 100 -> 200
        # We'll ensure each step exactly doubles the length.
        self.decoder = nn.Sequential(
            # (25 -> 50)
            nn.ConvTranspose1d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(0.1),
            
            # (50 -> 100)
            nn.ConvTranspose1d(in_channels=64, out_channels=32,
                               kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(0.1),
            
            # (100 -> 200)
            nn.ConvTranspose1d(in_channels=32, out_channels=input_channels,
                               kernel_size=4, stride=2, padding=1, output_padding=0)
            # => (B,3,200)
        )
        
    def encode(self, x):
        """
        Args:
            x: (B, 3, 200)
        Returns:
            z: (B, latent_dim)
        """
        x = self.encoder(x)         # (B,128,25)
        x = x.view(x.size(0), -1)   # (B, 128*25)
        z = self.enc_fc(x)          # (B, latent_dim)
        return z
    
    def decode(self, z):
        """
        Args:
            z: (B, latent_dim)
        Returns:
            x_recon: (B, 3, 200)
        """
        x = self.dec_fc(z)           # (B,128*25)
        x = x.view(x.size(0), 128, 25)  # (B,128,25)
        x_recon = self.decoder(x)    # (B,3,200)
        return x_recon
    
    def forward(self, x):
        """
        Standard forward for autoencoder training:
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

def train_autoencoder(ts_data_norm, device='cuda' if torch.cuda.is_available() else 'cpu', epochs=1000, batch_size=32):
    """Train the autoencoder with validation split and early stopping."""
    # Convert data to tensor
    ts_data_tensor = torch.tensor(ts_data_norm, dtype=torch.float32)
    
    # Create dataset & dataloader with validation split
    ae_dataset = TensorDataset(ts_data_tensor)
    train_size = int(0.9 * len(ae_dataset))
    val_size = len(ae_dataset) - train_size
    ae_train_ds, ae_val_ds = random_split(ae_dataset, [train_size, val_size])
    
    ae_train_loader = DataLoader(ae_train_ds, batch_size=batch_size, shuffle=True)
    ae_val_loader = DataLoader(ae_val_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize autoencoder
    latent_dim = 15
    model = Conv1DAutoencoder(input_channels=3, latent_dim=latent_dim).to(device)
    
    # Setup optimizer & loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    best_model = None
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for (x_batch,) in ae_train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_recon = model(x_batch)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_loss = epoch_train_loss / len(ae_train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for (x_batch,) in ae_val_loader:
                x_batch = x_batch.to(device)
                x_recon = model(x_batch)
                loss = criterion(x_recon, x_batch)
                epoch_val_loss += loss.item()
        
        val_loss = epoch_val_loss / len(ae_val_loader)
        val_losses.append(val_loss)
        
        print(f"[AE] Epoch {epoch+1}/{epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Autoencoder early stopping triggered.")
                model = best_model
                break
    
    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss (AE)')
    plt.plot(val_losses, label='Val Loss (AE)')
    plt.title('Autoencoder Reconstruction Loss')
    plt.legend()
    plt.savefig('vilar_plots/autoencoder_loss.png')
    plt.close()
    
    return model, (train_losses, val_losses)

def encode_dataset(model, ts_data_norm, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Encode the entire dataset to get latent representations."""
    model.eval()
    all_data = torch.tensor(ts_data_norm, dtype=torch.float32, device=device)
    latent_codes = []
    
    with torch.no_grad():
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i : i+batch_size]
            z_batch = model.encode(batch)
            latent_codes.append(z_batch.cpu().numpy())
    
    latent_codes = np.concatenate(latent_codes, axis=0)
    return latent_codes

def save_model_data(model, data_scaler, theta_scaler, summary_stats, filename):
    """Save the model, scalers and summary statistics."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'data_scaler': data_scaler,
        'theta_scaler': theta_scaler,
        'summary_stats': summary_stats
    }
    torch.save(save_dict, filename)

def load_model_data(model, filename):
    """Load the model, scalers and summary statistics."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['data_scaler'], checkpoint['theta_scaler'], checkpoint['summary_stats']
