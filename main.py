import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import sbibm
import copy
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import argparse
from utils import load_data, create_dataloaders, noise_schedule, betas_for_alpha_bar
import bayesflow as bf
from bayesflow import benchmarks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearConditionalDiffusionModel(nn.Module):
    def __init__(self, theta_dim, y_dim, layer_sizes, num_timesteps):
        super(LinearConditionalDiffusionModel, self).__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.num_timesteps = num_timesteps  # Store num_timesteps as a model attribute

        layers = []
        input_dim = theta_dim + y_dim + 1  # +1 for the time step t
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            #layers.append(nn.SiLU())
            layers.append(nn.ReLU())
            input_dim = size
        layers.append(nn.Linear(layer_sizes[-1], theta_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, theta, y, t):
        t_expanded = t.unsqueeze(-1).float() / self.num_timesteps
        input_data = torch.cat([theta, y, t_expanded], dim=-1)
        return self.net(input_data)

    def sample(self, N, y_observed):
        with torch.no_grad():
            intermediate_samples = []  # Store samples at different time steps
            theta_samples = torch.randn((N, self.theta_dim), device=device)
            for t in reversed(range(hyperparams['num_timesteps'])):
                if t % (hyperparams['num_timesteps'] // 10) == 0:
                    intermediate_samples.append(theta_samples.cpu().numpy())
                t_tensor = torch.full((N,), t, device=device)
                beta_t = self.beta[t].unsqueeze(-1).to(device)
                alpha_t = self.alpha[t].unsqueeze(-1).to(device)
                alpha_hat_t = self.alpha_hat[t].unsqueeze(-1).to(device)

                theta_pred = self.forward(theta_samples, y_observed, t_tensor)
                mean = (theta_samples - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t + 1e-5)) * theta_pred) / torch.sqrt(alpha_t + 1e-5)

                if t > 0:
                    noise = torch.randn_like(theta_samples)
                    theta_samples = mean + torch.sqrt(beta_t + 1e-5) * noise
                else:
                    theta_samples = mean

            intermediate_samples.append(theta_samples.cpu().numpy())
            return theta_samples, intermediate_samples

class CNNConditionalDiffusionModel(nn.Module):
    def __init__(self, theta_dim, y_dim, num_timesteps):
        super(CNNConditionalDiffusionModel, self).__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.num_timesteps = num_timesteps

        input_dim = theta_dim + y_dim + 1

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, theta_dim)
        )

    def forward(self, theta, y, t):
        t_expanded = t.unsqueeze(-1).float() / self.num_timesteps
        input_data = torch.cat([theta, y, t_expanded], dim=-1).unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        # Pass through Conv1D layers
        x = self.conv_layers(input_data)
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 128 * input_dim)
        # Pass through fully connected layers
        output = self.fc_layers(x)
        return output

    def sample(self, N, y_observed):
        with torch.no_grad():
            intermediate_samples = []  # Store samples at different time steps
            theta_samples = torch.randn((N, self.theta_dim), device=device)
            for t in reversed(range(self.num_timesteps)):
                if t % (self.num_timesteps // 10) == 0:
                    intermediate_samples.append(theta_samples.cpu().numpy())
                t_tensor = torch.full((N,), t, device=device)
                beta_t = self.beta[t].unsqueeze(-1).to(device)
                alpha_t = self.alpha[t].unsqueeze(-1).to(device)
                alpha_hat_t = self.alpha_hat[t].unsqueeze(-1).to(device)

                theta_pred = self.forward(theta_samples, y_observed, t_tensor)
                mean = (theta_samples - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t + 1e-5)) * theta_pred) / torch.sqrt(alpha_t + 1e-5)

                if t > 0:
                    noise = torch.randn_like(theta_samples)
                    theta_samples = mean + torch.sqrt(beta_t + 1e-5) * noise
                else:
                    theta_samples = mean

            intermediate_samples.append(theta_samples.cpu().numpy())
            return theta_samples, intermediate_samples



def diffusion_loss(model, theta_0, y, num_timesteps, mse_loss):
    t = torch.randint(0, num_timesteps, (theta_0.size(0),), device=device)
    noise = torch.randn_like(theta_0, device=device)
    alpha_hat_t = model.alpha_hat[t].unsqueeze(-1)
    theta_noisy = torch.sqrt(alpha_hat_t + 1e-5) * theta_0 + torch.sqrt(1 - alpha_hat_t + 1e-5) * noise
    theta_pred = model(theta_noisy, y, t)
    return mse_loss(theta_pred, noise)

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, num_timesteps):
    
    
    
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    best_model = None
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = diffusion_loss(model, theta_batch, y_batch, num_timesteps, mse_loss)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)
                val_loss += diffusion_loss(model, theta_batch, y_batch, num_timesteps, mse_loss).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= 20:
            #if counter >= hyperparams['patience']:
                print("Early stopping triggered.")
                model = best_model
                break
    
    
    return model, train_losses, val_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta_dim", type=int, default=5, help="Dimension of theta")
    parser.add_argument("--y_dim", type=int, default=8, help="Dimension of y")
    parser.add_argument("--num_timesteps", type=int, default=2000, help="Number of timesteps for diffusion")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Starting value of beta for noise schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending value of beta for noise schedule")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")  # Reduced batch size
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--beta_schedule", type=str, default='cosine', help="Type of beta schedule (linear, quadratic, cosine)")
    parser.add_argument("--layer_sizes", type=int, nargs='+', default=[512, 512, 512], help="Layer sizes for the neural network")
    parser.add_argument("--task_name", type=str, default="slcp", help="Task name for the experiment")
    parser.add_argument("--simulation_budgets", type=int, nargs='+', default=[10000, 20000, 30000], help="Simulation budgets to use")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for each simulation budget")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "cnn"], help="Type of model to use (linear or cnn)")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    M = 500  # Number of observations
    L = 250  # Number of posterior draws per observation

    # Define hyperparameters
    global hyperparams
    hyperparams = {
        'theta_dim': args.theta_dim,
        'y_dim': args.y_dim,
        'num_timesteps': args.num_timesteps,
        'learning_rate': args.learning_rate,
        'beta_start': args.beta_start,
        'beta_end': args.beta_end,
        'patience': args.patience,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'beta_schedule': args.beta_schedule,
        'layer_sizes': args.layer_sizes,
        'task_name': args.task_name,
        'simulation_budgets': args.simulation_budgets,
        'num_runs': args.num_runs
    }

    # Get simulation budgets and runs configuration
    simulation_budgets, num_runs = hyperparams['simulation_budgets'], hyperparams['num_runs']

    # Iterate over simulation budgets and runs
    for budget in simulation_budgets:
        for run in range(1, num_runs + 1):
            # Load data
            task_name = hyperparams['task_name']

            if task_name in ["lotka_volterra", "sir"]:
                benchmark = benchmarks.Benchmark(task_name)
                task = sbibm.get_task(task_name)

                def simulator(theta):
                    x = benchmark.generative_model.simulator(theta)
                    return x["sim_data"]

                def prior(N):
                    x = benchmark.generative_model.prior(N)
                    return x["prior_draws"]

                # Generate synthetic data
                num_samples = hyperparams['num_samples']
                theta_samples = prior(num_samples)
                y_data = simulator(theta_samples)

                # Scale the data
                theta_scaler = StandardScaler().fit(theta_samples)
                y_scaler = StandardScaler().fit(y_data)
                thetas = torch.tensor(theta_scaler.transform(theta_samples), dtype=torch.float32)
                y_data = torch.tensor(y_scaler.transform(y_data), dtype=torch.float32)

                # Create train and validation loaders
                dataset = TensorDataset(thetas, y_data)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

            else:
                thetas, y_data, theta_scaler, y_scaler, task = load_data(task_name, budget)
                train_loader, val_loader = create_dataloaders(thetas, y_data, hyperparams['batch_size'])

            # Initialize beta, alpha, and alpha_hat
            beta = noise_schedule(hyperparams['num_timesteps'], hyperparams['beta_start'], hyperparams['beta_end'], hyperparams['beta_schedule']).to(device)
            alpha = (1.0 - beta).to(device)
            alpha_hat = torch.cumprod(alpha, dim=0).to(device)

            # Choose the model based on the model_type argument
            if args.model_type == "linear":
                model = LinearConditionalDiffusionModel(
                    theta_dim=hyperparams['theta_dim'],
                    y_dim=hyperparams['y_dim'],
                    layer_sizes=hyperparams['layer_sizes'],
                    num_timesteps=hyperparams['num_timesteps']
                ).to(device)
            else:
                model = CNNConditionalDiffusionModel(
                    theta_dim=hyperparams['theta_dim'],
                    y_dim=hyperparams['y_dim'],
                    num_timesteps=hyperparams['num_timesteps']
                ).to(device)

            # Set beta, alpha, and alpha_hat attributes
            model.beta = beta
            model.alpha = alpha
            model.alpha_hat = alpha_hat

            # Optimizer and scheduler
            optimizer = optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            # Train the model
            start_time = time.time()
            model, train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, scheduler, hyperparams['num_epochs'], hyperparams['num_timesteps'])
            end_time = time.time()
            training_time = end_time - start_time

            # Print training time
            print(f"Training completed in {training_time/3600:.2f} hours ({training_time:.2f} seconds)")

            # Create results folder structure
            results_dir = os.path.join("results", task_name)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            training_info = {
                'model_state_dict': model.state_dict(),
                'training_time': training_time,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            # Save the model and training info
            model_filename = f"{task_name}_budget_{budget}_run_{run}.pth"
            model_path = os.path.join(results_dir, model_filename)
            torch.save(training_info, model_path)
            print(f"Model saved to {model_path}")
            
            # Convert training time to minutes
            training_time_minutes = training_time / 60
            
            # Save training time to txt file
            time_log_path = os.path.join(results_dir, "training_times.txt")
            with open(time_log_path, 'a') as f:
                f.write(f"Task: {task_name}\n")
                f.write(f"Simulation Budget: {budget}\n")
                f.write(f"Run: {run}\n")
                f.write(f"Training Time (minutes): {training_time_minutes:.2f}\n")
                f.write("-" * 50 + "\n")
            
            print(f"Training time: {training_time_minutes/60:.2f} hours ({training_time_minutes:.2f} minutes)")

            # Sampling posterior
            if task_name == 'sir':
                loaded_data = np.load('/diffusion/sir_task_data.npz', allow_pickle=True)
                true_param_1 = loaded_data['set_1'].item()['true_param']
                observed_data_1 = loaded_data['set_1'].item()['observed_data'][0]
                observed_data_1 = observed_data_1.reshape(1, -1)
                observed_data_1 = y_scaler.transform(observed_data_1)
                observed_data_1 = torch.tensor(observed_data_1, dtype=torch.float32)
                y_observed = observed_data_1.repeat(hyperparams['num_samples'], 1).to(device)
                theta_samples, intermediate_samples = model.sample(hyperparams['num_samples'], y_observed)
                theta_samples_descaled = theta_scaler.inverse_transform(theta_samples.cpu().numpy())
            else:
                y_observed = y_scaler.transform(np.array(task.get_observation(num_observation=1), dtype=np.float32))
                y_observed = torch.tensor(y_observed, dtype=torch.float32).repeat(hyperparams['num_samples'], 1).to(device)
                theta_samples, intermediate_samples = model.sample(hyperparams['num_samples'], y_observed)
                theta_samples_descaled = theta_scaler.inverse_transform(theta_samples.cpu().numpy())

            # Transform intermediate samples to the original scale
            intermediate_samples_descaled = [theta_scaler.inverse_transform(sample) for sample in intermediate_samples]

            # Save posterior samples
            posterior_filename = f"{task_name}_run_{run}_budget_{budget}.npz"
            posterior_path = os.path.join(results_dir, posterior_filename)
            np.savez(posterior_path, theta_samples=theta_samples_descaled)
            print(f"Posterior samples saved to {posterior_path}")

            # Save intermediate samples as a separate file
            intermediate_filename = f"{task_name}_intermediate_samples_run_{run}_budget_{budget}.npz"
            intermediate_path = os.path.join(results_dir, intermediate_filename)
            np.savez(intermediate_path, intermediate_samples=intermediate_samples_descaled)
            print(f"Intermediate samples saved to {intermediate_path}")

            ### SBC Diagnostics ###

            if task_name not in ["lotka_volterra", "sir"]:
                # For tasks other than "lotka_volterra" and "sir", use sbibm to get prior and simulator
                task = sbibm.get_task(task_name)
                prior = task.get_prior()
                simulator = task.get_simulator()
                theta_true = prior(M)  # Sample M true parameters from the prior
                y_data = simulator(theta_true)  # Generate observations in batch
            else:
                # Use existing prior and simulator functions for "lotka_volterra" and "sir"
                theta_true = prior(M)  # Sample M true parameters from the prior
                y_data = simulator(theta_true)  # Generate observations in batch

            # Scale SBC data directly without reshaping, as simulator now returns batch-compatible y_data
            y_data_scaled = torch.tensor(y_scaler.transform(y_data), dtype=torch.float32).to(device)
            theta_true_scaled = torch.tensor(theta_scaler.transform(theta_true), dtype=torch.float32).to(device)

            # Generate posterior samples for SBC
            posterior_draws = []
            for i in range(M):
                y_observed = y_data_scaled[i].unsqueeze(0).repeat(L, 1)
                theta_samples, _ = model.sample(L, y_observed)
                posterior_draws.append(theta_scaler.inverse_transform(theta_samples.cpu().numpy()))

            # Save SBC posterior samples and true theta
            sbc_filename = f"{task_name}_sbc_draws_budget_{budget}_run_{run}.npz"
            sbc_path = os.path.join(results_dir, sbc_filename)
            np.savez(sbc_path, posterior_draws=np.stack(posterior_draws), theta_true=theta_true)
            print(f"SBC posterior draws saved to {sbc_path}")



if __name__ == "__main__":
    main()
