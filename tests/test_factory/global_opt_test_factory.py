import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sggo.cluster import Cluster
from sggo.global_opt import GlobalOpt
from tests import utils


def create(global_opt: GlobalOpt, num_epochs: int):
    energy = global_opt.local_opt.energy

    class GlobalOptTestCase(unittest.TestCase):
        @settings(max_examples=5, deadline=None)
        @given(num_atoms=st.integers(min_value=1, max_value=32))
        def test_local_min_cpu(self, num_atoms: int):
            cluster_min = global_opt.find_minimum(num_atoms, num_epochs)
            cluster_opt = Cluster(np.array([]))
            cluster_opt.load(f"lj/{num_atoms}")

            utils.assert_cluster_on_cpu(self, cluster_min)
            self.assertEqual(cluster_min.positions.shape, (num_atoms, 3))
            self.assertEqual(cluster_min.positions.dtype, np.float32)

            self.assertTrue(np.close(energy.energy(cluster_min), energy.energy(cluster_opt), rtol=1e-3))

    return GlobalOptTestCase
