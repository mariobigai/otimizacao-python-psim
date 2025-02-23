import subprocess
import pandas as pd
import numpy as np
from scipy import integrate
from pyswarm import pso

# Define the simulation parameters
filename = "Buck - Controle PI_TF Vo.psimsch"
parameter_names = ['kp', 'ki', 'fc']
boundaries = [(0.1, 10), (100, 10000), (100, 10000)]
sim_failed = [0]


def run_PSIM(params):
    """
    Runs a PSIM simulation with the given parameters.
    """
    cmd = f'psimcmd -i {filename} -o "out.txt"'
    for name, value in zip(parameter_names, params):
        cmd += f' -v "{name}={value}"'
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode


def fitness_function(params):
    """
    Calculates the fitness of the given parameters.
    """
    if run_PSIM(params) == 0:
        # Read the simulation results
        df = pd.read_csv("out.txt", sep=r"\s+")
        time = df['Time']
        err = df['err']
        # Integrate the absolute error over time
        fitness = integrate.trapezoid(abs(err), time)
        print(f"Params: {params}, Fitness: {fitness}")
        return fitness
    else:
        sim_failed[0] += 1
        print(f"Params: {params}, Simulation failed.")
        return float('inf')  # Penalize failed simulations


# Bounds for the parameters
lower_bounds, upper_bounds = zip(*boundaries)

# Run the PSO optimization with verbosity in fitness_function
best_params, best_fitness = pso(
    fitness_function, 
    lower_bounds, 
    upper_bounds, 
    swarmsize=5, 
    maxiter=5, 
    minfunc=1e-6  # Convergence tolerance
)

# Print the results
print("\nBest Parameters:", {name: value for name, value in zip(parameter_names, best_params)})
print("Best Fitness:", best_fitness)
print("Number of Failed Simulations:", sim_failed[0])

# Visualize the best result
def plot_best(params):
    run_PSIM(params)
    df = pd.read_csv("out.txt", sep=r"\s+")
    time = df['Time']
    output = df['Vo']
    err = df['err']
    import matplotlib.pyplot as plt
    plt.plot(time, err)
    plt.title("Best Simulation Result")
    plt.xlabel("Time")
    plt.ylabel("Vo")
    plt.show()


plot_best(best_params)


