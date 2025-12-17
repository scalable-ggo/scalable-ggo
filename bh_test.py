from sggo.global_opt.basin_hopping import BasinHopping
from sggo.local_opt.fire import FIRECPU
from sggo.energy.lj import LJCPU
from numpy.typing import ArrayLike
import numpy as np

import sggo.visualize.plot as vis

def main():
    num_atoms = 17
    iterations = 100

    bh = BasinHopping(
        local_optimizer=FIRECPU(LJCPU()),
    )

    cluster = bh.find_minimum(num_atoms, iterations)[1]
    
    plot = vis.ClusterPlot(cluster)
    plot.plot()
    

if __name__ == "__main__":
    main()