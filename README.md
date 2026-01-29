# Scalable Global Geometry Optimization of Lennard-Jones Clusters

This project focuses on **accelerating material discovery** using **Global Geometry Optimization (GGO)** for atomic clusters. It is designed to efficiently find **low-energy clusters** of Lennard-Jones (LJ) clusters, which are commonly used as models in computational chemistry and nanomaterials research.

The Lennard-Jones potential describes the interaction between pairs of atoms:

V(r) = 4 * ε * [ (σ / r)^12 - (σ / r)^6 ]

where r is the distance, ε is the depth of the well, and σ is the distance at which the potential is zero.

---

## Features

- **Global Optimization Algorithms**: Supports methods such as Random Walk, Basin-Hopping and Genetic Algorithms.
- **Local Optimization Algorithms**: Includes implementations for FIRE, CG, and BFGS as local optimization algorithms.
- **Scalability**: Supports both NVIDIA GPU and MPI parallelization for maximum scalability.
- **Visualization Tools**: 3D plots of cluster structures.

---

## Installation

Requires an MPI implementation, **Python 3.13+** and standard python scientific libraries to function. In addition optional support for GPU based implementations require NVIDIA CUDA 13 toolkit:

```bash
git clone git@gitlab.ewi.tudelft.nl:kradziwilowicz/scalable-ggo.git
cd scalable-ggo
pip install -r requirements.txt
```

## Usage
The sggo module can be used standalone for custom python programs. In addition, a convenience script is provided to run a global optimization algorithm from a given configuration file. This script can be run with the example configuration as follows:
```bash
python3 sggo.py example.ini
```
All options supported by the configuration file are documented in the example configuration file example.ini. The script also supports being run with MPI. The following example uses mpirun to run the example configuration with 8 MPI processes:
```bash
mpirun -np 8 python3 sggo.py example.ini
```
