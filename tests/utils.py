from unittest import TestCase

import cupy as cp
import numpy as np
from hypothesis import strategies as st
import os
import unittest

from sggo.cluster import Cluster
from sggo.types import NDArray


@st.composite
def cpu_cluster(draw, max_size: int = 128) -> Cluster:
    n = draw(st.integers(min_value=1, max_value=max_size))
    # TODO: fix Hypothesis hanging when attempting to shrink examples with separation=0.2
    # position_rng = lambda lo, hi, size: np.array([draw(st.floats(min_value=lo, max_value=hi)) for _ in range(size)]).astype(np.float32)
    position_rng = None
    return Cluster.generate(n, rng=position_rng)


@st.composite
def gpu_cluster(draw, max_size: int = 128) -> Cluster:
    cluster = cpu_cluster(draw, max_size)
    cluster.positions = cp.asarray(cluster.positions)

    return cluster


def assert_cluster_on_cpu(testcase: TestCase, cluster: Cluster):
    testcase.assertIsInstance(cluster, Cluster)
    testcase.assertIsInstance(cluster.positions, np.ndarray)


def assert_cluster_on_gpu(testcase: TestCase, cluster: Cluster):
    testcase.assertIsInstance(cluster, Cluster)
    testcase.assertIsInstance(cluster.positions, cp.ndarray)


def assert_ndarray_on_cpu(testcase: TestCase, ndarray: NDArray):
    testcase.assertIsInstance(ndarray, np.ndarray)


def assert_ndarray_on_gpu(testcase: TestCase, ndarray: NDArray):
    testcase.assertIsInstance(ndarray, cp.ndarray)


skip_in_ci = unittest.skipIf("SGGO_CI" in os.environ or not cp.is_available(), "GPU not available")
