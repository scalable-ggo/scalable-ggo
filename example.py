from mpi4py import MPI

from sggo.energy import lj
from sggo.global_opt.basin_hopping import BasinHopping
from sggo.local_opt import bfgs, cg, fire
from sggo.visualize.plot import ClusterPlot

energy = lj.create()
local_opt = fire.create(energy)
BH = BasinHopping(local_opt)

energymin, clustermin = BH.find_minimum(128, 50)
if MPI.COMM_WORLD.rank == 0:
    print("Energy: ", energy.energy(clustermin))
    plot = ClusterPlot(clustermin)
    plot.plot()
