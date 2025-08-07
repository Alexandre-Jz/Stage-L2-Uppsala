import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestep_embedding(timesteps: torch.Tensor,
                       dim: int,
                       max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal time embedding, exactly as in noencod_vilar_diffusion."""
    if dim % 2 != 0:
        raise ValueError("t_embed_dim doit être pair")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    angles = timesteps.float().unsqueeze(1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def noise_schedule(num_timesteps: int,
                   beta_start: float = 1e-5,
                   beta_end: float = 5e-3,
                   schedule: str = "cosine"):
    """Identique à noencod_vilar_diffusion."""
    if schedule == "linear":
        beta = torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule == "quadratic":
        beta = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
    elif schedule == "cosine":
        beta = torch.tensor([
            (1 - math.cos((t / num_timesteps) * math.pi / 2)) ** 2
            for t in range(num_timesteps)
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_hat


class MLPDiffusionModel(nn.Module):
    def __init__(self,
                 theta_dim: int,
                 feature_dim: int = 15,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_timesteps: int = 500,
                 t_embed_dim: int = 64):
        super().__init__()
        self.theta_dim = theta_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.t_embed_dim = t_embed_dim

        # Projection d'entrée pour θ
        self.input_proj = nn.Linear(theta_dim, hidden_dim)

        # Blocs MLP résiduels
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                          nn.LeakyReLU(0.1))
            for _ in range(num_layers)
        ])

        # Prédit θ
        self.final = nn.Linear(hidden_dim, theta_dim)

        # FiLM network : prédit gamme & biais pour chaque bloc et chaque dimension cachée
        film_out = num_layers * hidden_dim * 2
        self.film_net = nn.Sequential(
            nn.Linear(feature_dim + t_embed_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, film_out)
        )

        # Buffers du schedule de diffusion
        beta, alpha, alpha_hat = noise_schedule(num_timesteps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)

    def forward(self,
                theta: torch.Tensor,  # (B, θ_dim)
                y: torch.Tensor,      # (B, feature_dim), déjà encodé
                t: torch.Tensor       # (B,) indices temporels
                ) -> torch.Tensor:
        # Embedding temporel
        t_emb = timestep_embedding(t, self.t_embed_dim)  # (B, t_embed_dim)
        # Condition concatenée
        cond = torch.cat([y, t_emb], dim=1)               # (B, feature_dim + t_embed_dim)

        # Générer gamma & beta
        film = self.film_net(cond).view(-1, self.num_layers, 2, self.hidden_dim)
        gammas = film[:, :, 0, :]   # (B, num_layers, hidden_dim)
        betas  = film[:, :, 1, :]   # (B, num_layers, hidden_dim)

        # Passer θ dans le MLP résiduel avec modulation FiLM
        h = self.input_proj(theta)
        for i, layer in enumerate(self.layers):
            h_prev = h
            h = layer(h)
            h = gammas[:, i] * h + betas[:, i]
            h = h + h_prev

        return self.final(h)

    @torch.no_grad()
    def sample(self, N: int, y: torch.Tensor, temperature: float = 1.0):
        """
        Génère N échantillons de θ conditionnés sur y via le processus inverse.
        y: (N, feature_dim)
        """
        theta = torch.randn((N, self.theta_dim), device=y.device) * temperature
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((N,), t, dtype=torch.long, device=y.device)
            a_hat = self.alpha_hat[t].unsqueeze(-1)
            a     = self.alpha[t].unsqueeze(-1)
            b     = self.beta[t].unsqueeze(-1)

            eps = self.forward(theta, y, t_tensor)
            denom = torch.sqrt(torch.clamp(1.0 - a_hat, min=1e-8))
            mu = (theta - (b / denom) * eps) / torch.sqrt(a)

            if t > 0:
                theta = mu + torch.sqrt(b) * torch.randn_like(theta) * temperature
            else:
                theta = mu
        return theta


def diffusion_loss(model: nn.Module,
                   theta_0: torch.Tensor,
                   y: torch.Tensor,
                   num_timesteps: int,
                   loss_fn: nn.Module) -> torch.Tensor:
    """DDPM loss, identique à noencod_vilar_diffusion."""
    B = theta_0.size(0)
    t = torch.randint(0, num_timesteps, (B,), device=theta_0.device)
    eps = torch.randn_like(theta_0)

    a_hat = model.alpha_hat[t].unsqueeze(-1)
    one_minus = torch.clamp(1.0 - a_hat, min=0.0)
    theta_noisy = torch.sqrt(a_hat) * theta_0 + torch.sqrt(one_minus) * eps
    eps_pred = model(theta_noisy, y, t)
    return loss_fn(eps_pred, eps)


def init_weights(m):
    """Xavier init for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def train_diffusion_model(model: nn.Module,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          num_epochs: int = 1000,
                          patience: int = 20,
                          learning_rate: float = 1e-3,
                          weight_decay: float = 1e-4,
                          lr_patience: int = 5,
                          lr_factor: float = 0.5,
                          min_lr: float = 1e-7):
    """Entraîne avec early stopping et ReduceLROnPlateau."""
    model.apply(init_weights)
    opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=lr_factor,
        patience=lr_patience, cooldown=5, min_lr=min_lr
    )
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    best_net = None
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # entraînement
        model.train()
        running = 0.0
        for θ, y in train_loader:
            θ, y = θ.to(device), y.to(device)
            loss = diffusion_loss(model, θ, y, model.num_timesteps, loss_fn)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        avg_tr = running / len(train_loader)
        train_losses.append(avg_tr)

        # validation
        model.eval()
        running = 0.0
        with torch.no_grad():
            for θ, y in val_loader:
                θ, y = θ.to(device), y.to(device)
                running += diffusion_loss(model, θ, y, model.num_timesteps, loss_fn).item()
        avg_va = running / len(val_loader)
        val_losses.append(avg_va)

        scheduler.step(avg_va)
        lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:4d} | train {avg_tr:.4f} | val {avg_va:.4f} | lr {lr:.2e}")

        # early stopping
        if avg_va < best_val:
            best_val, best_net, wait = avg_va, copy.deepcopy(model), 0
        else:
            wait += 1
            if wait >= patience:
                print("★ Early-stopping")
                break

    # tracer pertes
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
    plt.tight_layout(); plt.savefig('diffusion_training.pdf'); plt.close()

    return best_net, train_losses, val_losses


def create_dataloaders(theta: np.ndarray,
                       y: np.ndarray,
                       batch_size: int = 64,
                       train_split: float = 0.9):
    """
    θ: np.ndarray (N, θ_dim)
    y: np.ndarray (N, 15) vecteurs passés par votre autoencodeur
    """
    θ = torch.from_numpy(theta).float()
    y = torch.from_numpy(y).float()
    dataset = TensorDataset(θ, y)
    n_train = int(train_split * len(dataset))
    train_ds, val_ds = random_split(
        dataset,
        [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(42)
    )
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size))
