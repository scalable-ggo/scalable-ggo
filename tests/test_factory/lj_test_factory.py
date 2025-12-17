import unittest

import cupy as cp
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from hypothesis import given

from sggo.cluster import Cluster
from sggo.energy.lj import LJ
from tests import utils


def cluster_to_aselj(cluster: Cluster):
    return Atoms(
        positions=cp.asnumpy(cluster.positions),
        calculator=LennardJones(rc=np.inf)
    )


def create(lj: LJ):
    class LJTestCase(unittest.TestCase):
        @given(cluster=utils.cpu_cluster())
        def test_energy_agrees(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy = lj.energy(cluster)
            energy_ase = cluster_to_aselj(cluster).get_potential_energy()

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_cpu(self, energy)
            self.assertTrue(np.allclose(energy, energy_ase, rtol=1e-3))

        @given(cluster=utils.cpu_cluster())
        def test_energies_agrees(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energies = lj.energies(cluster)
            energies_ase = cluster_to_aselj(cluster).get_potential_energies()

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_cpu(self, energies)
            self.assertTrue(np.allclose(energies, energies_ase, rtol=1e-3))

        @given(cluster=utils.cpu_cluster())
        def test_energy_gradient_agrees(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy_gradient = lj.energy_gradient(cluster)
            energy_gradient_ase = -cluster_to_aselj(cluster).get_forces()

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_cpu(self, energy_gradient)
            self.assertTrue(np.allclose(energy_gradient, energy_gradient_ase, rtol=1e-3))

        @unittest.skipIf(not cp.is_available(), "GPU not available")
        @given(cluster=utils.gpu_cluster())
        def test_energy_agrees_gpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy = lj.energy(cluster)
            energy_ase = cp.asarray(cluster_to_aselj(cluster).get_potential_energy())

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_gpu(self, energy)
            self.assertTrue(cp.allclose(energy, energy_ase, rtol=1e-3))

        @unittest.skipIf(not cp.is_available(), "GPU not available")
        @given(cluster=utils.gpu_cluster())
        def test_energies_agrees_gpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energies = lj.energies(cluster)
            energies_ase = cp.asarray(cluster_to_aselj(cluster).get_potential_energies())

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_gpu(self, energies)
            self.assertTrue(cp.allclose(energies, energies_ase, rtol=1e-3))

        @unittest.skipIf(not cp.is_available(), "GPU not available")
        @given(cluster=utils.gpu_cluster())
        def test_energy_gradient_agrees_gpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy_gradient = lj.energy_gradient(cluster)
            energy_gradient_ase = cp.asarray(-cluster_to_aselj(cluster).get_forces())

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_gpu(self, energy_gradient)
            self.assertTrue(cp.allclose(energy_gradient, energy_gradient_ase, rtol=1e-3))

    return LJTestCase
