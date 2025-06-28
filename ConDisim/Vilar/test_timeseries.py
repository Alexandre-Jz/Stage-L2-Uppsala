import numpy as np
import matplotlib.pyplot as plt
from vilar_dataset import simulator, parameter_names, Vilar_Oscillator, SSACSolver
import os

# Initialize vilar_model and vilar_solver
vilar_model = Vilar_Oscillator()
vilar_solver = SSACSolver(model=vilar_model)

# Function to generate synthetic time series data using the simulator
def generate_time_series(true_params):
    # Use simulator to generate time series data
    time_series_data = simulator(true_params, model=vilar_model, solver=vilar_solver, transform=True)
    return time_series_data

# Function to plot the time series data
def plot_time_series(time_series_data):
    # Plot settings
    species_names = ['Complex (C)', 'Activator (A)', 'Repressor (R)']
    colors = ['#2E86C1', '#E74C3C', '#2ECC71']  # Professional color palette
    time = np.arange(time_series_data.shape[-1])
    
    # Create figure with specific size for two-column paper
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    plt.subplots_adjust(hspace=0.4)  # Increase space between subplots for labels
    
    for i, (ax, species, color) in enumerate(zip(axes, species_names, colors)):
        # Plot first trajectory (darker)
        main_line = ax.plot(time, time_series_data[0, i], color=color, linewidth=1.5,
                           label='Reference', zorder=5)[0]
        
        # Plot additional trajectories (lighter)
        other_line = None
        for j in range(1, len(time_series_data)):
            line = ax.plot(time, time_series_data[j, i], alpha=0.3, color=color,
                          linewidth=0.8, label='Stochastic Realizations' if j == 1 else None)
            if j == 1:
                other_line = line[0]
        
        # Customize each subplot
        ax.set_ylabel('Population', fontsize=12)
        
        # Place species name
        ax.text(-0.15, 0.5, species, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='center')
        
        # Only show x-label for bottom plot
        if i == 2:
            ax.set_xlabel('Time', fontsize=12)
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add grid but make it very subtle
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend aligned with species name
        leg = ax.legend([main_line, other_line], ['Reference', 'Stochastic Realizations'],
                       fontsize=10, loc='upper right',
                       bbox_to_anchor=(0.98, 1.2),
                       frameon=True, framealpha=0.9,
                       ncol=2)  # Two columns, horizontal layout
        
        # Make legend lines thicker (only in legend)
        leg_lines = leg.get_lines()
        for line in leg_lines:
            line.set_linewidth(4)  # Adjust this number to change thickness
        
        leg.get_frame().set_linewidth(0.5)
    
    # Create directory if it doesn't exist
    os.makedirs('vilar_plots', exist_ok=True)
    
    # Save figure with high DPI for publication quality
    plt.savefig('vilar_plots/time_series_3x1.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved as 'vilar_plots/time_series_3x1.pdf'")

# Main function to execute the test
def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 5 trajectories with the same parameters
    true_params = np.array([50, 500, 0.01, 50, 50, 5, 10, 0.5, 1, 0.2, 1, 1, 2, 50, 100]).reshape(1,-1)
    trajectories = []
    
    # Generate 5 trajectories
    for _ in range(5):
        time_series = generate_time_series(true_params)
        trajectories.append(time_series[0])  # Remove the batch dimension
    
    # Stack all trajectories
    all_trajectories = np.stack(trajectories)
    print(f"Generated {len(trajectories)} trajectories, shape: {all_trajectories.shape}")
    
    # Plot the trajectories
    plot_time_series(all_trajectories)

if __name__ == "__main__":
    main()
