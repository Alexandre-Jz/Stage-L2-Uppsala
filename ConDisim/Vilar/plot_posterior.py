import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for matplotlib (no display required)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from vilar_dataset import simulator, Vilar_Oscillator
from gillespy2 import SSACSolver

def load_results(results_dir, budget, run_num):

    results_path = os.path.join('posterior_samples', f'vilar_posterior_{budget}_run_{run_num}.npz')
    dataset_path = os.path.join('datasets', f'vilar_dataset_{budget}.npz')
    
    print(f"\nAttempting to load:")
    print(f"Results file: {results_path}")
    print(f"Dataset file: {dataset_path}")
    
    if not os.path.exists(results_path) or not os.path.exists(dataset_path):
        print(f"Warning: One or more required files not found:")
        if not os.path.exists(results_path):
            print(f"- Missing results file: {results_path}")
        if not os.path.exists(dataset_path):
            print(f"- Missing dataset file: {dataset_path}")
        return None, None, None
    
    try:
        # Load posterior samples
        print("Loading posterior samples...")
        results = np.load(results_path, allow_pickle=True)
        posterior_samples = results['posterior_samples']
        print(f"Posterior samples shape: {posterior_samples.shape}")
        
        # Load true parameters and data
        print("Loading dataset...")
        data = np.load(dataset_path, allow_pickle=True)
        true_params = data['true_theta']
        observed_data = data['true_ts']
        print(f"Observed data shape: {observed_data.shape}")
        
        return true_params, posterior_samples, observed_data
    except Exception as e:
        print(f"Warning: Error loading data: {str(e)}")
        return None, None, None

def plot_posterior_distributions(true_params, posterior_samples, save_path, budget, run_num):
    """Create KDE plots for posterior distributions."""
    param_names = [
        r'$\alpha_a$', r'$\alpha_a^\prime$', r'$\alpha_r$', 
        r'$\alpha_r^\prime$', r'$\beta_a$', r'$\beta_r$', 
        r'$\delta_{ma}$', r'$\delta_{mr}$', r'$\gamma_a$',
        r'$\gamma_r$', r'$\gamma_c$', r'$\theta_a$', 
        r'$\theta_r$', r'$\delta_a$', r'$\delta_r$'
    ]

    # Create a figure with 3x5 subplots
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))

    # Define prior ranges
    dmin = [0, 100, 0, 20, 10, 1, 1, 0, 0, 0, 0.5, 0, 0, 0, 0]
    dmax = [80, 600, 4, 60, 60, 7, 12, 2, 3, 0.7, 2.5, 4, 3, 70, 300]
    prior_ranges = list(zip(dmin, dmax))

    # First create all plots to get max density for normalization
    densities = []
    for i in range(15):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Get KDE estimate
        kde = stats.gaussian_kde(posterior_samples[:, i])
        x_range = np.linspace(prior_ranges[i][0], prior_ranges[i][1], 200)
        density = kde(x_range)
        densities.append(density)

    # Now create the actual plots with normalized densities
    for i in range(15):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
            
        dmin, dmax = prior_ranges[i]
        
        # Normalize the density
        kde = stats.gaussian_kde(posterior_samples[:, i])
        x_range = np.linspace(dmin, dmax, 200)
        density = kde(x_range)
        density = density / density.max()  # Normalize to [0,1]
        
        # Plot normalized density
        ax.fill_between(x_range, density, color='blue', alpha=0.6, label='Posterior')
        
        # True value line
        ax.axvline(true_params[i], color='red', linestyle='--', linewidth=2, label='True')
        
        # Prior range
        ax.axvspan(dmin, dmax, color='gray', alpha=0.2, label='Prior')
        
        # Set y-axis limits and ticks with clean formatting
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        # Remove the original tick labels
        ax.set_yticklabels(['', '1'])
        # Add custom positioned 0 label aligned with other ticks
        ax.text(-0.05, 0.1, '0', transform=ax.get_yaxis_transform(),
                verticalalignment='center', horizontalalignment='right', fontsize=20)
        
        # Format x-axis ticks
        def format_ticks(x, p):
            if abs(x) < 0.01:  # For very small numbers
                if x == 0:
                    return ''
                return f'{x:.3f}'  # Show small numbers with more precision
            return f'{x:g}'
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
        
        # Move y-axis label
        ax.yaxis.set_label_coords(-0.2, 0.6)
        
        # Set x-axis limits to prior range and show only 3 ticks
        ax.set_xlim(dmin, dmax)
        ax.set_xticks([dmin, true_params[i], dmax])
        
        # Remove x-label and place parameter name above plot
        ax.set_xlabel('')
        ax.text(0.5, 1.1, param_names[i], fontsize=30,
                ha='center', va='bottom', transform=ax.transAxes)
        
        if col == 0:  # Only add Density label to first plot in each row
            ax.set_ylabel('Density', fontsize=30)
        else:
            ax.set_ylabel('')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make tick labels larger
        ax.tick_params(axis='both', which='major', labelsize=20)

    # Create a single legend at the top of the figure with a box
    handles, labels = axes[0,0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', 
              bbox_to_anchor=(0.5, 1.13), ncol=3, 
              fontsize=40, frameon=True)
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')

    # Hide any remaining subplots in the last row
    for row in range(3):
        for col in range(5):
            if row * 5 + col >= 15:
                axes[row, col].set_visible(False)

    plt.tight_layout()
    # Adjust layout to make room for the legend
    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.95)  # Increased hspace for parameter names
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()



def plot_posterior_predictive(posterior_samples, observed_data, save_path):
    """Plot posterior predictive checks comparing simulated vs observed data."""
    # Initialize model and solver
    model = Vilar_Oscillator()
    solver = SSACSolver(model=model)
    
    # Select random subset of posterior samples for simulation
    np.random.seed(42)  # For reproducibility
    num_samples = 3
    random_indices = np.random.choice(len(posterior_samples), num_samples, replace=False)
    selected_samples = posterior_samples[random_indices]
    
    # Simulate trajectories for selected samples
    simulated_data = []
    for params in selected_samples:
        sim_result = simulator(params[np.newaxis, :], model, solver)
        sim_result = sim_result.squeeze(0)
        simulated_data.append(sim_result)
    simulated_data = np.array(simulated_data)
    
    # Plot settings
    species_names = ['Complex (C)', 'Activator (A)', 'Repressor (R)']
    colors = ['#2E86C1', '#E74C3C', '#2ECC71']  # Professional color palette
    time = np.arange(observed_data.shape[1])
    
    # Create figure with specific size for two-column paper
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    plt.subplots_adjust(hspace=0.4)  # Increase space between subplots for labels
    
    for i, (ax, species, color) in enumerate(zip(axes, species_names, colors)):
        # Plot observed data first (so it appears first in legend)
        true_line = ax.plot(time, observed_data[i], color=color, linewidth=1.5,
                           label='True', zorder=5)[0]
        
        # Plot simulated trajectories
        gen_line = None
        for j in range(num_samples):
            line = ax.plot(time, simulated_data[j, i], alpha=0.3, color=color, 
                          linewidth=0.8, label='Generated' if j == 0 else None)
            if j == 0:
                gen_line = line[0]
        
        # Customize each subplot
        ax.set_ylabel('Population', fontsize=12)
        
        # Place species name
        ax.text(0.3, 1.05, species, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center')
        
        # Only show x-label for bottom plot
        if i == 2:
            ax.set_xlabel('Time', fontsize=12)
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add grid but make it very subtle
        ax.grid(True, linestyle='--', alpha=0.3, color='gray' )
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        
        # Add legend aligned with species name
        leg = ax.legend([true_line, gen_line], ['True', 'Generated'],
                       fontsize=10, loc='upper center',
                       bbox_to_anchor=(0.8, 1.2),
                       frameon=True, framealpha=0.9,
                       ncol=2)  # Two columns, horizontal layout
        
        # Make legend lines thicker (only in legend)
        leg_lines = leg.get_lines()
        for line in leg_lines:
            line.set_linewidth(4)  # Adjust this number to change thickness
        
        leg.get_frame().set_linewidth(0.5)
    
    # Save figure with high DPI for publication quality
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate plots for all budgets and runs."""
    print("\nStarting plot generation...")
    
    # Create output directory
    os.makedirs('vilar_plots', exist_ok=True)
    print("Created output directory: vilar_plots/")
    
    # Set parameters
    budgets = [10000, 20000, 30000]  # Single budget
    runs = range(1, 2)  # Single run
    
    successful_plots = False  # Track if any plots were generated
    
    for budget in budgets:
        for run in runs:
            print(f"\nProcessing budget {budget}, run {run}")
            print("-" * 50)
            
            # Load results
            true_params, posterior_samples, observed_data = load_results('.', budget, run)
            
            # Skip if data not available
            if true_params is None or posterior_samples is None or observed_data is None:
                print(f"Skipping budget {budget}, run {run} due to missing data")
                continue
            
            try:
                # Generate posterior distribution plots
                print("\nGenerating posterior distribution plots...")
                plot_posterior_distributions(
                    true_params,
                    posterior_samples,
                    f'vilar_plots/posterior_distributions_budget_{budget}_run_{run}.pdf',
                    budget,
                    run
                )
                
                # Generate posterior predictive plots
                print("\nGenerating posterior predictive plots...")
                plot_posterior_predictive(
                    posterior_samples,
                    observed_data,
                    f'vilar_plots/posterior_predictive_budget_{budget}_run_{run}.pdf'
                )
                
                successful_plots = True
                print(f"\nSuccessfully generated all plots for budget {budget}, run {run}")
                print("Output files:")
                print(f"1. vilar_plots/posterior_distributions_budget_{budget}_run_{run}.pdf")
                print(f"2. vilar_plots/posterior_predictive_budget_{budget}_run_{run}.pdf")
            
            except Exception as e:
                print(f"\nError generating plots for budget {budget}, run {run}:")
                print(f"Error details: {str(e)}")
                continue
    
    if not successful_plots:
        print("\nWarning: No plots were generated. All data combinations were unavailable.")
    else:
        print("\nPlot generation complete!")

if __name__ == "__main__":
    main()
