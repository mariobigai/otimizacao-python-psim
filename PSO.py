import subprocess
import pandas as pd
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

def run_PSIM(position):
	cmd = 'psimcmd -i ' + filename + ' -o "out.txt"'
	for i in range(dimension):
		cmd += ' -v "' + parameter_names[i] + '=' + str(position[i]) + '"'
	return subprocess.run(cmd, capture_output=True).returncode

def calculate_fitness(position):
	if run_PSIM(position) == 0:
		df = pd.read_csv("out.txt", sep=r"\s+")
		time = df['Time']
		err = df['err']
		return integrate.trapezoid(abs(err), time)
	else:
		sim_failed[0] += 1
		return float('inf')

def calculate_SD(swarm):
	position_med = []
	position_SD = []
	for i in range(dimension):
		position_med.append(0)
		position_SD.append(0)
	for p in swarm:
		for i in range(dimension):
			position_med[i] += p.Actual_Pos[i]/len(swarm)		
	for p in swarm:
		for i in range(dimension):
			position_SD[i] += ((p.Actual_Pos[i] - position_med[i])**2)/len(swarm)
	for i in range(dimension):
		position_SD[i] = position_SD[i]/gBestPos[i]
	return np.sqrt(position_SD)

def plot_best(position):
	run_PSIM(position)
	df = pd.read_csv("out.txt", sep=r"\s+")
	x = df['Time']
	y = df['Vo']
	plt.plot(x, y)
	plt.show()

class particle:
	w = 0.5
	cognitive = 0.5
	social = 0.5
	def __init__(self):
		self.Actual_Pos = []
		self.velocity = []
		self.Best_Pos = []
		for i in range(dimension):
			self.Actual_Pos.append(boundaries[i][0] + (boundaries[i][1]-boundaries[i][0])*random.random())
			self.velocity.append((boundaries[i][1]-boundaries[i][0])*(1-2*random.random()))
			self.Best_Pos.append(self.Actual_Pos[i])
		self.pBest = calculate_fitness(self.Actual_Pos)

filename = "Buck - Controle PI_TF Vo.psimsch"
dimension = 3
parameter_names = ['kp', 'ki', 'fc']
boundaries = [[0.1, 10], [100, 10000], [100, 10000]]
maxiter = 5
popsize = 5
gBest = float('inf')
gBestPos = []
for i in range(dimension):
	gBestPos.append(float('inf'))
swarm = []
sim_failed = [0]

if len(boundaries) != dimension:
	print('The dimension of the array "boundaries" must be equal to the dimension of the problem')
	exit()

if len(parameter_names) != dimension:
	print('The dimension of the array "parameter_names" must be equal to the dimension of the problem')
	exit()

for i in range(popsize):
	print("\rInitializing Swarm: " + str(i+1) + "/" + str(popsize), end='')
	p = particle()
	if p.pBest < gBest:
		gBest = p.pBest
		for n in range(dimension):
			gBestPos[n] = p.Best_Pos[n]
	swarm.append(p)

print("")

for i in range(maxiter):
	print("\rIteration: " + str(i+1) + "/" + str(maxiter), end='')
	for p in swarm:
		for n in range(dimension):
			p.velocity[n] = p.w*p.velocity[n] + p.cognitive*random.random()*(p.Best_Pos[n]-p.Actual_Pos[n]) + p.social*random.random()*(gBestPos[n]-p.Actual_Pos[n])
			p.Actual_Pos[n] += p.velocity[n]
		for n in range(dimension):	
			if p.Actual_Pos[n] > boundaries[n][1]:
				p.Actual_Pos[n] = boundaries[n][1]
				p.velocity[n] = -p.w*p.velocity[n]
			elif p.Actual_Pos[n] < boundaries[n][0]:
				p.Actual_Pos[n] = boundaries[n][0]
				p.velocity[n] = -p.w*p.velocity[n]
		fitness = calculate_fitness(p.Actual_Pos)
		if fitness < p.pBest:
			for n in range(dimension):
				p.Best_Pos[n] = p.Actual_Pos[n]
			p.pBest = fitness
			if p.pBest < gBest:
				gBest = p.pBest
				for n in range(dimension):
					gBestPos[n] = p.Best_Pos[n]

print("")

str_best = ""
for i in range(dimension):
	str_best += str(parameter_names[i]) + '=' + str(gBestPos[i]) + ' '
print('Best: ' + str_best)
print('Fitness: ' + str(gBest))
str_SD = ""
Std_Dev = calculate_SD(swarm)
for i in range(dimension):
	str_SD += '\u03C3_' + str(parameter_names[i]) + '=' + str(Std_Dev[i]) + ' '
print('Standard Deviation: ' + str_SD)
print('Number of Failed Simulations: ' + str(sim_failed[0]))
plot_best(gBestPos)