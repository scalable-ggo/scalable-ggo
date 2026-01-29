#!/usr/bin/env python3
import configparser
import os
import sys

from mpi4py import MPI

from sggo.energy import lj
from sggo.global_opt.basin_hopping import BasinHopping
from sggo.global_opt.genetic import GeneticAlgorithm
from sggo.local_opt import bfgs, cg, fire
from sggo.visualize.plot import ClusterPlot


def main():
    argc = len(sys.argv)

    if argc != 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(f"Usage: {sys.argv[0]} config_file.ini")
        if argc != 2:
            os.exit(1)
        else:
            os.exit(0)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    epochs = config.getint("global_opt", "epochs", fallback=None)
    size = config.getint("general", "cluster_size", fallback=None)
    target = config.getfloat("general", "target", fallback=None)

    global_opt_alg = config.get("global_opt", "algorithm", fallback="RandomWalk")
    local_opt_alg = config.get("local_opt", "algorithm", fallback="FIRE")
    local_opt_imp = config.get("local_opt", "implementation", fallback="CPU")
    lj_imp = config.get("energy.lj", "implementation", fallback=local_opt_imp)
    out_file = config.get("output", "file", fallback=None)

    if epochs is None:
        print("The number of epochs is required")
        os.exit(1)

    if size is None:
        print("The cluster size is required")
        os.exit(1)

    energy = None
    local_opt = None
    global_opt = None

    match lj_imp.upper():
        case "CPU":
            energy = lj.create(variant=lj.LJVariant.CPU)
        case "GPU":
            energy = lj.create(variant=lj.LJVariant.GPU)
        case "GPUKERNEL":
            energy = lj.create(variant=lj.LJVariant.GPUKERNEL)
        case _:
            print(f"Unkown LJ implementation {lj_imp}")
            os.exit(1)

    match local_opt_alg.upper():
        case "FIRE":
            match local_opt_imp.upper():
                case "CPU":
                    local_opt = fire.create(energy, variant=fire.FIREVariant.CPU)
                case "GPU":
                    local_opt = fire.create(energy, variant=fire.FIREVariant.GPU)
                case "GPUKERNEL":
                    local_opt = fire.create(energy, variant=fire.FIREVariant.GPUKERNEL)
                case _:
                    print(f"Unkown FIRE implementation {local_opt_alg}")
                    os.exit(1)
        case "CG":
            match local_opt_imp.upper():
                case "CPU":
                    local_opt = cg.create(energy, variant=cg.CGVariant.CPU)
                case "GPU":
                    local_opt = cg.create(energy, variant=cg.CGVariant.GPU)
                case _:
                    print(f"Unkown CG implementation {local_opt_alg}")
                    os.exit(1)
        case "BFGS":
            match local_opt_imp.upper():
                case "CPU":
                    local_opt = bfgs.create(energy, variant=bfgs.BFGSVariant.CPU)
                case "GPU":
                    local_opt = bfgs.create(energy, variant=bfgs.BFGSVariant.GPU)
                case _:
                    print(f"Unkown BFGS implementation {local_opt_alg}")
                    os.exit(1)
        case _:
            print(f"Unkown global optimization algorithm {local_opt_alg}")
            os.exit(1)

    match global_opt_alg.upper():
        case "BASINHOPPING":
            global_opt = BasinHopping(local_opt)
        case "GENETICALGORITHM":
            global_opt = GeneticAlgorithm(10, local_opt, operators = [1,2,3,4,5])
        case _:
            print(f"Unkown global optimization algorithm {local_opt_alg}")
            os.exit(1)

    energy, clustermin = global_opt.find_minimum(num_atoms=size, num_epochs=epochs, target=target)
    if MPI.COMM_WORLD.rank == 0:
        print("Energy of the best cluster: ", energy)
        plot = ClusterPlot(clustermin)
        plot.plot()

        if out_file is not None:
            clustermin.save(out_file)

if __name__ == "__main__":
    main()
