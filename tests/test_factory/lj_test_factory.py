import unittest

import cupy as cp
import numpy as np
import numpy.testing as npt
from ase import Atoms
from ase.calculators.lj import LennardJones
from hypothesis import given, settings

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
        @settings(max_examples=50, deadline=None)
        @given(cluster=utils.cpu_cluster())
        def test_energy_agrees(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy = lj.energy(cluster)
            energy_ase = cluster_to_aselj(cluster).get_potential_energy()

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_cpu(self, energy)
            self.assertEqual(energy.shape, (1,))
            self.assertEqual(energy.dtype, np.float32)
            npt.assert_allclose(energy, energy_ase, rtol=1e-3)

        @settings(max_examples=50, deadline=None)
        @given(cluster=utils.cpu_cluster())
        def test_energies_agrees(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energies = lj.energies(cluster)
            energies_ase = cluster_to_aselj(cluster).get_potential_energies()

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_cpu(self, energies)
            self.assertEqual(energies.shape, (len(cluster.positions),))
            self.assertEqual(energies.dtype, np.float32)
            npt.assert_allclose(energies, energies_ase, rtol=1e-3)

        @settings(max_examples=50, deadline=None)
        @given(cluster=utils.cpu_cluster())
        def test_energy_gradient_agrees(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy_gradient = lj.energy_gradient(cluster)
            energy_gradient_ase = -cluster_to_aselj(cluster).get_forces()

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_cpu(self, energy_gradient)
            self.assertEqual(energy_gradient.shape, (len(cluster.positions), 3))
            self.assertEqual(energy_gradient.dtype, np.float32)
            npt.assert_allclose(energy_gradient, energy_gradient_ase, rtol=1e-3)

        @utils.skip_in_ci
        @settings(max_examples=50, deadline=None)
        @given(cluster=utils.gpu_cluster())
        def test_energy_agrees_gpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy = lj.energy(cluster)
            energy_ase = cp.asarray(cluster_to_aselj(cluster).get_potential_energy())

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_gpu(self, energy)
            self.assertEqual(energy.shape, (1,))
            self.assertEqual(energy.dtype, cp.float32)
            self.assertTrue(cp.allclose(energy, energy_ase, rtol=1e-3))

        @utils.skip_in_ci
        @settings(max_examples=50, deadline=None)
        @given(cluster=utils.gpu_cluster())
        def test_energies_agrees_gpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energies = lj.energies(cluster)
            energies_ase = cp.asarray(cluster_to_aselj(cluster).get_potential_energies())

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_gpu(self, energies)
            self.assertEqual(energies.shape, (len(cluster.positions),))
            self.assertEqual(energies.dtype, cp.float32)
            self.assertTrue(cp.allclose(energies, energies_ase, rtol=1e-3))

        @utils.skip_in_ci
        @settings(max_examples=50, deadline=None)
        @given(cluster=utils.gpu_cluster())
        def test_energy_gradient_agrees_gpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()

            energy_gradient = lj.energy_gradient(cluster)
            energy_gradient_ase = cp.asarray(-cluster_to_aselj(cluster).get_forces())

            self.assertEqual(cluster, cluster_backup)
            utils.assert_ndarray_on_gpu(self, energy_gradient)
            self.assertEqual(energy_gradient.shape, (len(cluster.positions), 3))
            self.assertEqual(energy_gradient.dtype, cp.float32)
            self.assertTrue(cp.allclose(energy_gradient, energy_gradient_ase, rtol=1e-3))

    return LJTestCase
