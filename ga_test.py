from mpi4py import MPI

from sggo.energy import lj
from sggo.global_opt.genetic import GeneticAlgorithm
from sggo.local_opt import bfgs, cg, fire
from sggo.visualize.plot import ClusterPlot

energy = lj.create()
local_opt = fire.create(energy)
ga = GeneticAlgorithm(num_candidates=10, local_optimizer=local_opt, operators=[1,2,3,4,5])

energymin, clustermin = ga.find_minimum(128, 50)
if MPI.COMM_WORLD.rank == 0:
    print("Energy: ", energymin)
    plot = ClusterPlot(clustermin)
    plot.plot()
