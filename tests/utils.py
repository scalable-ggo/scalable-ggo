from unittest import TestCase

import cupy as cp
import numpy as np
from hypothesis import strategies as st

from sggo.cluster import Cluster
from sggo.types import NDArray


@st.composite
def cpu_cluster(draw, max_size: int = 128) -> Cluster:
    n = draw(st.integers(min_value=1, max_value=max_size))
    return Cluster.generate(n)


@st.composite
def gpu_cluster(draw, max_size: int = 128) -> Cluster:
    n = draw(st.integers(min_value=1, max_value=max_size))
    cluster = Cluster.generate(n)
    cluster.positions = cp.asarray(cluster.positions)

    return cluster


def assert_cluster_on_cpu(testcase: TestCase, cluster: Cluster):
    testcase.assertIsInstance(cluster.positions, np.ndarary)


def assert_cluster_on_gpu(testcase: TestCase, cluster: Cluster):
    testcase.assertIsInstance(cluster.positions, cp.ndarary)


def assert_ndarray_on_cpu(testcase: TestCase, ndarray: NDArray):
    testcase.assertIsInstance(ndarray, np.ndarray)


def assert_ndarray_on_gpu(testcase: TestCase, ndarray: NDArray):
    testcase.assertIsInstance(ndarray, cp.ndarray)
