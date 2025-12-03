from numpy.typing import ArrayLike
import cupy as cp


def assimilar(source: ArrayLike, target: ArrayLike) -> ArrayLike:
    if isinstance(target, cp.ndarray) or hasattr(target, "__cuda_array_interface__"):
        return cp.asarray(source)
    else:
        return cp.asnumpy(source)
