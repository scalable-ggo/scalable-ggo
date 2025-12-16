import unittest

import numpy as np
from sggo.cluster import Cluster
from sggo.global_opt.genetic import GeneticAlgorithm


class TestGenetic(unittest.TestCase):
    def test_mate(self):
        ga = GeneticAlgorithm(2, None, lambda: 0)
        p1 = np.array([
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1]
        ])
        a = np.sqrt(2)
        p2 = (np.array([
            [a, 0, -a],
            [0, 1, 0],
            [a, 0, a]
        ]) @ p1.T).T
        c = ga.mate(Cluster(p1), Cluster(p2))
        self.assertEqual(len(c.positions), len(p1))
        all_points = np.concat([p1, p2])
        for p in c.positions:
            self.assertIn(p, all_points)
