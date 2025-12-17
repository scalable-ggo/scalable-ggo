import os
import unittest

import cupy as cp
import numpy as np
from hypothesis import given, settings

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt
from tests import utils


def create(local_opt: LocalOpt):
    energy = local_opt.energy
    tol = 1e-1

    class LocalOptTestCase(unittest.TestCase):
        @settings(max_examples=10, deadline=None)
        @given(cluster=utils.cpu_cluster(max_size=64))
        def test_local_min_cpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()
            cluster_min = local_opt.local_min(cluster, target_gradient=tol, max_steps=1000)

            self.assertEqual(cluster, cluster_backup)

            utils.assert_cluster_on_cpu(self, cluster_min)
            self.assertEqual(cluster_min.positions.shape, cluster.positions.shape)
            self.assertEqual(cluster_min.positions.dtype, np.float32)

            self.assertLessEqual(energy.energy(cluster_min), energy.energy(cluster))
            self.assertLess(np.linalg.norm(energy.energy_gradient(cluster_min), axis=1).max(), tol)
            if len(cluster.positions) > 1:
                self.assertLess(np.max(energy.energies(cluster_min)), -0.5 + 5e-4)

        @unittest.skipIf(not cp.is_available() or "CI" in os.environ, "GPU not available")
        @settings(max_examples=10, deadline=None)
        @given(cluster=utils.gpu_cluster(max_size=64))
        def test_local_min_gpu(self, cluster: Cluster):
            cluster_backup = cluster.deepcopy()
            cluster_min = local_opt.local_min(cluster, target_gradient=tol, max_steps=1000)

            self.assertEqual(cluster, cluster_backup)

            utils.assert_cluster_on_gpu(self, cluster_min)
            self.assertEqual(cluster_min.positions.shape, cluster.positions.shape)
            self.assertEqual(cluster_min.positions.dtype, cp.float32)

            self.assertLessEqual(energy.energy(cluster_min), energy.energy(cluster))
            self.assertLess(cp.linalg.norm(energy.energy_gradient(cluster_min), axis=1).max(), tol)
            if len(cluster.positions) > 1:
                self.assertLess(cp.max(energy.energies(cluster_min)), -0.5 + 5e-4)

    return LocalOptTestCase
