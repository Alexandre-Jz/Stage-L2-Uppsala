task_names = [

    'slcp_distractors',

]

# Iterate over all specified task names
for task_name in task_names:
    task_path = os.path.join(results_dir, task_name)
    if os.path.isdir(task_path):  # Check if it is a directory

        # Create the directory for storing C2ST results for the task if it does not exist
        task_results_dir = os.path.join(c2st_results_dir, task_name)
        if not os.path.exists(task_results_dir):
            os.makedirs(task_results_dir)

        # Path to the consolidated JSON file for this task
        result_file = os.path.join(task_results_dir, f'{task_name}_c2st_results.json')

        # Load existing results if the JSON file already exists
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                c2st_results = json.load(f)
        else:
            c2st_results = {}

        # Load the true posterior samples for the current task
        task = sbibm.get_task(task_name)
        true_posterior = task.get_reference_posterior_samples(num_observation=1)

        # Convert true posterior to PyTorch tensor
        true_posterior = torch.tensor(true_posterior.numpy(), dtype=torch.float32)

        # Iterate over all .npz files for each task
        for file_name in os.listdir(task_path):
            # Only process files that match the desired format and skip 'intermediate' and 'sbc_draws'
            if (file_name.endswith('.npz') and 'run' in file_name and 'budget' in file_name
                and 'intermediate' not in file_name and 'sbc_draws' not in file_name):

                file_path = os.path.join(task_path, file_name)
                print(f"Processing file: {file_path}")

                # Extract simulation budget and run number from the file name
                parts = file_name.split('_')
                if 'budget' in parts and 'run' in parts:
                    budget_index = parts.index('budget')
                    run_index = parts.index('run')

                    # Extract the simulation budget and run number, removing the '.npz' extension
                    simulation_budget = int(parts[budget_index + 1].replace('.npz', ''))
                    run_number = int(parts[run_index + 1].replace('.npz', ''))

                    # Check if this result already exists in the JSON data
                    if str(simulation_budget) in c2st_results and str(run_number) in c2st_results[str(simulation_budget)]:
                        print(f"Result for {file_name} already exists, skipping...")
                        continue

                    # Load the generated samples from the file
                    data = np.load(file_path)
                    theta_samples = data['theta_samples']

                    # Convert generated samples to PyTorch tensor
                    theta_samples = torch.tensor(theta_samples, dtype=torch.float32)

                    # Compute C2ST metric
                    c2st_value = c2st(true_posterior, theta_samples).item()  # Convert tensor to a Python float

                    # Store the result in the nested dictionary
                    if str(simulation_budget) not in c2st_results:
                        c2st_results[str(simulation_budget)] = {}
                    c2st_results[str(simulation_budget)][str(run_number)] = {
                        'c2st_accuracy': c2st_value
                    }

                    print(f"Added C2ST result for budget {simulation_budget}, run {run_number}")

        # Save the updated C2ST results to the JSON file
        with open(result_file, 'w') as f:
            json.dump(c2st_results, f, indent=4)

        print(f"C2ST results for task '{task_name}' saved to {result_file}")