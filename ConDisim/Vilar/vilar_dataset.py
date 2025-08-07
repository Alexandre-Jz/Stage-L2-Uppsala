import traceback
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import random
from collections.abc import Mapping
import multiprocessing as mp
import time
import copy
import gillespy2
from sklearn.preprocessing import MinMaxScaler
from gillespy2 import SSACSolver
from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from gillespy2 import EventAssignment, EventTrigger, Event
from gillespy2.core.events import *
from functools import partial
from vilar_autoencoder import Conv1DAutoencoder, train_autoencoder, encode_dataset

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define parameter names
parameter_names = ['alpha_a', 'alpha_a_prime', 'alpha_r', 'alpha_r_prime', 'beta_a', 
                  'beta_r', 'delta_ma', 'delta_mr', 'delta_a', 'delta_r', 'gamma_a', 
                  'gamma_r', 'gamma_c', 'theta_a', 'theta_r']

class Vilar_Oscillator(gillespy2.Model):
    def __init__(self, parameter_values=None):
        gillespy2.Model.__init__(self, name="Vilar_Oscillator")
        self.volume = 1

        # Parameters
        self.add_parameter(gillespy2.Parameter(name="alpha_a", expression=50))
        self.add_parameter(gillespy2.Parameter(name="alpha_a_prime", expression=500))
        self.add_parameter(gillespy2.Parameter(name="alpha_r", expression=0.01))
        self.add_parameter(gillespy2.Parameter(name="alpha_r_prime", expression=50))
        self.add_parameter(gillespy2.Parameter(name="beta_a", expression=50))
        self.add_parameter(gillespy2.Parameter(name="beta_r", expression=5))
        self.add_parameter(gillespy2.Parameter(name="delta_ma", expression=10))
        self.add_parameter(gillespy2.Parameter(name="delta_mr", expression=0.5))
        self.add_parameter(gillespy2.Parameter(name="delta_a", expression=1))
        self.add_parameter(gillespy2.Parameter(name="delta_r", expression=0.2))
        self.add_parameter(gillespy2.Parameter(name="gamma_a", expression=1))
        self.add_parameter(gillespy2.Parameter(name="gamma_r", expression=1))
        self.add_parameter(gillespy2.Parameter(name="gamma_c", expression=2))
        self.add_parameter(gillespy2.Parameter(name="theta_a", expression=50))
        self.add_parameter(gillespy2.Parameter(name="theta_r", expression=100))

        # Species
        self.add_species(gillespy2.Species(name="Da", initial_value=1, mode="discrete"))
        self.add_species(gillespy2.Species(name="Da_prime", initial_value=0, mode="discrete"))
        self.add_species(gillespy2.Species(name="Ma", initial_value=0, mode="discrete"))
        self.add_species(gillespy2.Species(name="Dr", initial_value=1, mode="discrete"))
        self.add_species(gillespy2.Species(name="Dr_prime", initial_value=0, mode="discrete"))
        self.add_species(gillespy2.Species(name="Mr", initial_value=0, mode="discrete"))
        self.add_species(gillespy2.Species(name="C", initial_value=10, mode="discrete"))
        self.add_species(gillespy2.Species(name="A", initial_value=10, mode="discrete"))
        self.add_species(gillespy2.Species(name="R", initial_value=10, mode="discrete"))

        # Reactions
        self.add_reaction(gillespy2.Reaction(name="r1", reactants={'Da_prime': 1}, products={'Da': 1}, rate=self.listOfParameters["theta_a"]))
        self.add_reaction(gillespy2.Reaction(name="r2", reactants={'Da': 1, 'A': 1}, products={'Da_prime': 1}, rate=self.listOfParameters["gamma_a"]))
        self.add_reaction(gillespy2.Reaction(name="r3", reactants={'Dr_prime': 1}, products={'Dr': 1}, rate=self.listOfParameters["theta_r"]))
        self.add_reaction(gillespy2.Reaction(name="r4", reactants={'Dr': 1, 'A': 1}, products={'Dr_prime': 1}, rate=self.listOfParameters["gamma_r"]))
        self.add_reaction(gillespy2.Reaction(name="r5", reactants={'Da_prime': 1}, products={'Da_prime': 1, 'Ma': 1}, rate=self.listOfParameters["alpha_a_prime"]))
        self.add_reaction(gillespy2.Reaction(name="r6", reactants={'Da': 1}, products={'Da': 1, 'Ma': 1}, rate=self.listOfParameters["alpha_a"]))
        self.add_reaction(gillespy2.Reaction(name="r7", reactants={'Ma': 1}, products={}, rate=self.listOfParameters["delta_ma"]))
        self.add_reaction(gillespy2.Reaction(name="r8", reactants={'Ma': 1}, products={'A': 1, 'Ma': 1}, rate=self.listOfParameters["beta_a"]))
        self.add_reaction(gillespy2.Reaction(name="r9", reactants={'Da_prime': 1}, products={'Da_prime': 1, 'A': 1}, rate=self.listOfParameters["theta_a"]))
        self.add_reaction(gillespy2.Reaction(name="r10", reactants={'Dr_prime': 1}, products={'Dr_prime': 1, 'A': 1}, rate=self.listOfParameters["theta_a"]))
        self.add_reaction(gillespy2.Reaction(name="r11", reactants={'A': 1}, products={}, rate=self.listOfParameters["gamma_c"]))
        self.add_reaction(gillespy2.Reaction(name="r12", reactants={'A': 1, 'R': 1}, products={'C': 1}, rate=self.listOfParameters["gamma_c"]))
        self.add_reaction(gillespy2.Reaction(name="r13", reactants={'Dr_prime': 1}, products={'Dr_prime': 1, 'Mr': 1}, rate=self.listOfParameters["alpha_r_prime"]))
        self.add_reaction(gillespy2.Reaction(name="r14", reactants={'Dr': 1}, products={'Dr': 1, 'Mr': 1}, rate=self.listOfParameters["alpha_r"]))
        self.add_reaction(gillespy2.Reaction(name="r15", reactants={'Mr': 1}, products={}, rate=self.listOfParameters["delta_mr"]))
        self.add_reaction(gillespy2.Reaction(name="r16", reactants={'Mr': 1}, products={'Mr': 1, 'R': 1}, rate=self.listOfParameters["beta_r"]))
        self.add_reaction(gillespy2.Reaction(name="r17", reactants={'R': 1}, products={}, rate=self.listOfParameters["delta_r"]))
        self.add_reaction(gillespy2.Reaction(name="r18", reactants={'C': 1}, products={'R': 1}, rate=self.listOfParameters["delta_a"]))

        # Timespan
        self.timespan(np.linspace(0, 200, 200))

def simulator(params, model, solver, transform=True):
    'Simulator takes parameter vector as input and return a time series data of shape 3x200'
    params = params.ravel()
    res = model.run(
        solver=solver,
        timeout=0.5,
        variables={parameter_names[i]: params[i] for i in range(len(parameter_names))}
    )

    if res.rc == 33:
        return np.ones((1, 3, 200))

    if transform:
        # Extract only observed species
        sp_C = res['C']
        sp_A = res['A']
        sp_R = res['R']
        return np.vstack([sp_C, sp_A, sp_R])[np.newaxis, :, :]
    else:
        return res

def generate_data_parallel(N, model, solver):
    """ Returns the parameters and TS data using parallel processing"""
    dmin = [0,    100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0] #lower parameter boundary
    dmax = [80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300] #upper parameter boundary
    params = np.random.uniform(low=dmin, high=dmax, size=(N,15))
    num_cores = mp.cpu_count()  # Gets the number of available CPU cores
    
    # Create a partial function with both model and solver fixed
    simulator_with_model = partial(simulator, model=model, solver=solver)
    
    with mp.Pool(processes=num_cores) as pool:
        ts = pool.map(simulator_with_model, params)
    ts = np.asarray(ts)  # Shape should be (N, 1, 3, 200)
    return ts, params  # Return time series first, then parameters

def save_dataset():
    # Create necessary directories
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('vilar_plots', exist_ok=True)

    # Initialize model and solver
    model = Vilar_Oscillator()
    solver = SSACSolver(model=model)
    
    # Generate 30k dataset
    print("\nGenerating dataset with 30,000 simulations...")
    ts_data, theta = generate_data_parallel(30000, model, solver)
    print(f"Generated data shapes:")
    print(f"Time series data: {ts_data.shape}")
    print(f"Parameters: {theta.shape}")
    
    # Get dimensions
    N, _, C, T = ts_data.shape  # Shape: (N, 1, 3, 200)
    ts_data = ts_data.squeeze(1)  # Remove singleton dimension: (N, 3, 200)
    print(f"Squeezed time series data: {ts_data.shape}")

    # Initialize scalers
    time_series_scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(C)]
    theta_scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize time series data channel-wise
    ts_data_norm = np.zeros_like(ts_data)
    for i in range(C):
        # Reshape for scaler: (N, T) => "N samples, T features"
        ts_data_norm[:, i, :] = time_series_scalers[i].fit_transform(ts_data[:, i, :])
    print(f"Normalized time series data: {ts_data_norm.shape}")

    # Normalize parameters
    theta_norm = theta_scaler.fit_transform(theta)
    print(f"Normalized parameters: {theta_norm.shape}")

    # Train autoencoder and generate embeddings
    print("\nTraining autoencoder...")
    autoencoder = Conv1DAutoencoder(input_channels=3, latent_dim=15).to(device)
    autoencoder, loss_history = train_autoencoder(ts_data_norm, device=device)
    print(f"Loss history length: {len(loss_history)}")
    
    # Generate embeddings for the entire dataset
    print("\nGenerating embeddings...")
    ts_embeddings = encode_dataset(autoencoder, ts_data_norm, device=device)
    print(f"Time series embeddings: {ts_embeddings.shape}")

    # Generate reference trajectory with true parameters
    print("\nGenerating reference trajectory...")
    true_theta = np.array([50, 500, 0.01, 50, 50, 5, 10, 0.5, 1, 0.2, 1, 1, 2, 50, 100])
    true_ts = simulator(true_theta, model, solver)
    true_ts = true_ts.squeeze(0)  # Remove batch dimension
    print(f"True parameters: {true_theta.shape}")
    print(f"True time series: {true_ts.shape}")
    
    # Normalize reference trajectory
    true_ts_scaled = np.zeros_like(true_ts)
    for i in range(C):
        true_ts_scaled[i, :] = time_series_scalers[i].transform(true_ts[i, :].reshape(1, -1))
    print(f"Normalized true time series: {true_ts_scaled.shape}")
    
    # Generate embedding for reference trajectory
    true_ts_embedding = encode_dataset(autoencoder, true_ts_scaled[np.newaxis], device=device)
    print(f"True time series embedding: {true_ts_embedding.shape}")

    # Save the full 30k dataset
    print("\nSaving datasets...")
    for size in [20000]:
        # Take the first 'size' samples for each subset
        subset_idx = slice(0, size)
        
        save_dict = {
            'theta': theta[subset_idx],
            'ts_data': ts_data[subset_idx],
            'theta_norm': theta_norm[subset_idx],
            'ts_data_norm': ts_data_norm[subset_idx],
            'ts_embeddings': ts_embeddings[subset_idx],
            'true_theta': true_theta,
            'true_ts': true_ts,
            'true_ts_scaled': true_ts_scaled,
            'true_ts_embedding': true_ts_embedding,
            'time_series_scalers': time_series_scalers,
            'theta_scaler': theta_scaler
        }
        
        dataset_path = f'datasets/vilar_dataset_{size}.npz'
        np.savez_compressed(dataset_path, **save_dict)
        print(f"\nSaved dataset with {size} samples to {dataset_path}")
        print(f"Subset shapes:")
        print(f"- theta: {theta[subset_idx].shape}")
        print(f"- ts_data: {ts_data[subset_idx].shape}")
        print(f"- theta_norm: {theta_norm[subset_idx].shape}")
        print(f"- ts_data_norm: {ts_data_norm[subset_idx].shape}")
        print(f"- ts_embeddings: {ts_embeddings[subset_idx].shape}")


if __name__ == "__main__":
    save_dataset()