from sggo.energy import lj
from sggo.global_opt.random_walk import RandomRestartBaseline
from sggo.local_opt import bfgs, cg, fire
from sggo.visualize.plot import ClusterPlot

energy = lj.create()
local_opt = fire.create(energy)

baseline = RandomRestartBaseline(local_optimizer=local_opt)

best_cluster, best_energy, hist = baseline.find_minimum(
    num_atoms=3,
    num_trials=10000,     # fair comparison: same number of local_min calls as GA evaluations
    seed=0,
    return_history=True
)

print(best_energy)

plot = ClusterPlot(best_cluster)
plot.plot()
