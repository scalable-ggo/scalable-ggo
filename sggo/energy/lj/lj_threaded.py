import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
from numpy.typing import ArrayLike

from sggo.cluster import Cluster
from sggo.energy.lj import LJ


class LJThreaded(LJ):
    _thread_pool = None
    _thread_pool_lock = threading.Lock()
    _n_threads = None

    def __init__(self, n_threads=None) -> None:
        super().__init__()

        self._get_thread_pool(n_threads=n_threads)

    @classmethod
    def _get_thread_pool(cls, n_threads=None):
        if n_threads is None:
            n_threads = os.cpu_count()

        with cls._thread_pool_lock:
            if cls._thread_pool is None or cls._n_threads != n_threads:
                if cls._thread_pool is not None:
                    cls._thread_pool.shutdown(wait=True)
                cls._thread_pool = ThreadPoolExecutor(max_workers=n_threads)
                cls._n_threads = n_threads
        return cls._thread_pool

    @classmethod
    def shutdown_thread_pool(cls):
        with cls._thread_pool_lock:
            if cls._thread_pool is not None:
                cls._thread_pool.shutdown(wait=True)
                cls._thread_pool = None
                cls._n_threads = None

    def energies(self, cluster: Cluster) -> ArrayLike:
        pos = cp.asnumpy(cluster.positions)

        n_atoms = pos.shape[0]
        pool = self._get_thread_pool()

        n_threads = self._n_threads
        chunk_size = max(1, n_atoms // n_threads)

        chunks = []
        for i in range(0, n_atoms, chunk_size):
            end = min(i + chunk_size, n_atoms)
            chunks.append((i, end))

        futures = [pool.submit(_compute_energies_chunk, pos, start, end) for start, end in chunks]

        energies = np.empty(n_atoms, dtype=np.float32)
        for future, (start, end) in zip(futures, chunks):
            energies[start:end] = future.result()

        return energies

    def energy_gradient(self, cluster: Cluster) -> ArrayLike:
        pos = cp.asnumpy(cluster.positions)

        n_atoms = pos.shape[0]
        n_dims = pos.shape[1]
        pool = self._get_thread_pool()

        n_threads = self._n_threads
        chunk_size = max(1, n_atoms // n_threads)

        chunks = []
        for i in range(0, n_atoms, chunk_size):
            end = min(i + chunk_size, n_atoms)
            chunks.append((i, end))

        futures = [pool.submit(_compute_gradient_chunk, pos, start, end) for start, end in chunks]

        gradients = np.empty((n_atoms, n_dims), dtype=np.float32)
        for future, (start, end) in zip(futures, chunks):
            gradients[start:end] = future.result()

        return gradients


def _compute_energies_chunk(pos: np.ndarray, start: int, end: int) -> np.ndarray:
    n_atoms = pos.shape[0]
    chunk_energies = np.zeros(end - start, dtype=np.float32)

    for idx, i in enumerate(range(start, end)):
        energy = 0.0
        for j in range(n_atoms):
            if i == j:
                continue

            disp = pos[i] - pos[j]
            r2 = np.sum(disp * disp)

            c2 = 1.0 / r2
            c6 = c2 * c2 * c2
            c12 = c6 * c6
            energy += 2.0 * (c12 - c6)

        chunk_energies[idx] = energy

    return chunk_energies


def _compute_gradient_chunk(pos: np.ndarray, start: int, end: int) -> np.ndarray:
    n_atoms = pos.shape[0]
    n_dims = pos.shape[1]
    chunk_gradients = np.zeros((end - start, n_dims), dtype=np.float32)

    for idx, i in enumerate(range(start, end)):
        gradient = np.zeros(n_dims, dtype=np.float32)

        for j in range(n_atoms):
            if i == j:
                continue

            disp = pos[i] - pos[j]
            r2 = np.sum(disp * disp)

            c2 = 1.0 / r2
            c6 = c2 * c2 * c2
            c12 = c6 * c6
            mag = -24.0 * (2.0 * c12 - c6) * c2

            gradient += mag * disp

        chunk_gradients[idx] = gradient

    return chunk_gradients
