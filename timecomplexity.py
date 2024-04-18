import numpy as np
import matplotlib.pyplot as plt

# Set the font size for all plot elements
plt.rcParams.update({'font.size': 8})

def simulate_grid_search(num_params, values_per_param):
    # Grid search explores all combinations
    return values_per_param ** num_params

def simulate_random_search(num_iterations):
    # Random search performs a fixed number of iterations
    return num_iterations

def simulate_halving_grid_search(num_params, values_per_param):
    # Halving grid search reduces the number of combinations in log steps
    total = 0
    while values_per_param > 1:
        total += values_per_param ** num_params
        values_per_param //= 2
    return total

def simulate_halving_random_search(num_params, values_per_param, base_iterations=10):
    # Halving random search starts with a random subset and halves
    total = base_iterations
    while values_per_param > 1:
        total += base_iterations * (values_per_param ** num_params)
        values_per_param //= 2
    return total

def plot_time_complexities():
    params = np.arange(1, 6)  # 1 to 5 parameters
    values_per_param = 10  # Each parameter can take 10 different values
    iterations = 100  # Fixed number of iterations for random search
    
    grid_search_results = [simulate_grid_search(p, values_per_param) for p in params]
    random_search_results = [simulate_random_search(iterations) for _ in params]
    halving_grid_search_results = [simulate_halving_grid_search(p, values_per_param) for p in params]
    halving_random_search_results = [simulate_halving_random_search(p, values_per_param, base_iterations=10) for p in params]

    plt.figure(figsize=(10, 6))
    plt.plot(params, grid_search_results, label='Grid Search', marker='o')
    plt.plot(params, random_search_results, label='Random Search', marker='o')
    plt.plot(params, halving_grid_search_results, label='Halving Grid Search', marker='o')
    plt.plot(params, halving_random_search_results, label='Halving Random Search', marker='o')
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Number of Combinations/Iterations')
    plt.legend()
    plt.grid(True)

    ax = plt.gca()  # Get current axes

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid
    ax.grid(True, linestyle='--', color='grey', alpha=0.4)

    plt.show()

plot_time_complexities()