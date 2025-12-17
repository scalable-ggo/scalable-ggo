import cProfile

from sggo.cluster import Cluster
from sggo.energy import lj
from sggo.global_opt import basin_hopping
from sggo.local_opt import bfgs, cg, fire
from sggo.visualize import plot


energy = lj.create()
local_opt = fire.create(energy)
cluster = Cluster.generate(128)

BH = basin_hopping.BasinHopping(local_opt)

emin, clustermin = None, None
BH.find_minimum(cluster, 2000)
cProfile.run("emin, clustermin = BH.find_minimum(cluster, 2000)", sort=1)

print("Energy: ", emin)
plot = plot.ClusterPlot(clustermin)
plot.plot()
