import numpy as np # Arrays
from scipy import integrate # Usado para calcular a integral do erro
import subprocess # Usado para rodar o PSIM
import pandas as pd # Usado para ler o arquivo de saída do PSIM
import pyswarms as ps # Importa PySwarms
### Display das partículas
from IPython.display import Image
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
###

# Define the simulation parameters
filename = "Bridgeless_Boost_PFC_Feedforward.psimsch" #"Buck - Controle PI_TF Vo.psimsch"
parameter_names = ['kp', 'ki', 'fc']
boundaries = [(0.1, 1000), (0.1, 100), (100, 10000)]
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
    fit = np.ones(len(params))
    for i in range(len(fit)):
        if run_PSIM(params[i]) == 0:
            # Read the simulation results
            df = pd.read_csv("out.txt", sep=r"\s+")
            time = df['Time']
            err = df['err']
            # Integrate the absolute error over time
            fitness = integrate.trapezoid(abs(err), time)
            print(f"Params: {params[i]}, Fitness: {fitness}")
            fit[i] = fitness
        else:
            sim_failed[0] += 1
            print(f"Params: {params[i]}, Simulation failed.")
            fit[i] = float(1e10) #float('inf') ocasionamnete causa erro
    return fit

# Muda formato da matriz das fronteiras, para ficar assim: ex: ([0.1, 1, 1], [1000, 1000, 100000])
bounds = (np.array([b[0] for b in boundaries]), np.array([b[1] for b in boundaries]))

# Parâmetros do Swarm
options = {'c1':0.5, 'c2':0.3, 'w':0.9}

# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=4, dimensions=3, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(fitness_function, iters=2)

##########################
# Plota resultados

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example pos_history from optimizer (replace with your actual pos_history)
pos_history = optimizer.pos_history  # Shape: (iterations, swarm_size, dimensions)

# Convert to numpy array for easier slicing
pos_history = np.array(pos_history)

fig, ax = plt.subplots()
sc = ax.scatter([], [])

def update(frame):
    ax.clear()
    ax.scatter(pos_history[frame][:, 0], pos_history[frame][:, 1])
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    #ax.set_xlim(np.min(pos_history[:,:,0]), np.max(pos_history[:,:,0]))  # Adjust limits based on data
    #ax.set_ylim(np.min(pos_history[:,:,1]), np.max(pos_history[:,:,1]))  
    ax.set_title(f"Iteration {frame}")

ani = animation.FuncAnimation(fig, update, frames=len(pos_history), repeat=False)
ani.save('swarm.gif', writer='imagemagick', fps=5)
Image(url='swarm.gif')
#ax.clear()

##########################

print("Number of Failed Simulations:", sim_failed[0])

# Visualize the best result
def plot_best(params):
    run_PSIM(params)
    df = pd.read_csv("out.txt", sep=r"\s+")
    time = df['Time']
    I_L = df['I(Lin1)']
    plt.figure()
    plt.plot(time, I_L)
    plt.title("Best Simulation Result")
    plt.xlabel("Time")
    plt.ylabel("I(Lin1)")
    plt.show()

plot_best(pos)