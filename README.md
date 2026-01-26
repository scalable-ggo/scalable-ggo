# Scalable Global Geometry Optimization of Lennard-Jones Clusters

This project focuses on **accelerating material discovery** using **Global Geometry Optimization (GGO)** for atomic clusters. It is designed to efficiently find **low-energy clusters** of Lennard-Jones (LJ) clusters, which are commonly used as models in computational chemistry and nanomaterials research.

The Lennard-Jones potential describes the interaction between pairs of atoms:

V(r) = 4 * ε * [ (σ / r)^12 - (σ / r)^6 ]

where r is the distance, ε is the depth of the well, and σ is the distance at which the potential is zero.

---

## Features

- **Global Optimization Algorithms**: Supports methods such as Basin-Hopping and Genetic Algorithms.
- **Separation Enforcement**: Ensures a minimum distance between atoms to prevent overlaps.
- **Visualization Tools**: 3D plots of cluster structures.
- **Scalability**: Optimized for clusters for large clusters using GPU programming.

---

## Installation

Requires **Python 3.13+** and standard scientific libraries:

```bash
git clone git@gitlab.ewi.tudelft.nl:kradziwilowicz/scalable-ggo.git
cd scalable-ggo
pip install -r requirements.txt

