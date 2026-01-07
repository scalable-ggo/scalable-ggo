import cProfile

from sggo.cluster import Cluster
from sggo.energy import lj
from sggo.global_opt.basin_hopping import BasinHopping
from sggo.local_opt import bfgs, cg, fire
from sggo.visualize.plot import ClusterPlot

energy = lj.create()
local_opt = fire.create(energy)
cluster = Cluster.generate(128)

BH = BasinHopping(local_opt)

emin, clustermin = None, None
cProfile.run("emin, clustermin = BH.find_minimum(cluster, 50)", sort=1)

print("Energy: ", emin)
plot = ClusterPlot(clustermin)
plot.plot()
