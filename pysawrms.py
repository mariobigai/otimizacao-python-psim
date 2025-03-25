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

# controlador proporcional resonante - fotovoltaica



############################# Variáveis
resumeProgress = False
n_particles = 50
iters = 20
filename = "Bridgeless_Boost_PFC_Feedforward.psimsch" #"Buck - Controle PI_TF Vo.psimsch"
parameter_names = ['kp', 'ki', 'fc']
boundaries = [(0.01, 0.05), (20, 400), (1000, 20000)] # [(0.01, 1), (10, 1000), (1000, 20000)]
sim_failed = [0]

# Parâmetros do Swarm
options = {'c1':2, 'c2':1.7, 'w':0.9}
#############################

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



# Call instance of PSO with bounds argument
if resumeProgress:
    # Load best swarm positions from file (ensure shape: (n_particles, dimensions))
    init_pos = np.loadtxt("best.txt")
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=3, options=options, bounds=bounds,init_pos=init_pos)
else:
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=3, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(fitness_function, iters=iters)

print("Number of Failed Simulations:", sim_failed[0])

# Save best swarm positions for future use as init_pos
np.savetxt("best.txt", optimizer.swarm.position)

########################## - Pré-requisitos para plotagem

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example pos_history from optimizer (replace with your actual pos_history)
pos_history = optimizer.pos_history  # Shape: (iterations, swarm_size, dimensions)
fit_history = optimizer.cost_history  # Shape: (iterations,)

########################## - 2D Plot

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

########################## - Best result plot

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
    #plt.show() - deixa p/ plotar no fim do código (impede travamento)

plot_best(pos)

############################# - 3D Plot

# Create a figure and a 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()  # Clear previous points
    # Plot the current iteration's positions in 3D space
    ax.scatter(pos_history[frame][:, 0], pos_history[frame][:, 1], fit_history[frame])
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_zlim(0, max(fit_history))
    ax.set_title(f"Iteration {frame}")

# Create the animation; frames equals the number of iterations
ani = animation.FuncAnimation(fig, update, frames=len(pos_history), repeat=False)
ani.save('swarm3d.gif', writer='imagemagick', fps=20)
Image(url='swarm3d.gif')

# Set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Fitness vs kp e ki')


# Mostra todos os plots
plt.show()