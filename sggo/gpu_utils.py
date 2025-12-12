import cupy as cp

from sggo.types import NDArray


def assimilar(source: NDArray, target: NDArray) -> NDArray:
    if isinstance(target, cp.ndarray) or hasattr(target, "__cuda_array_interface__"):
        return cp.asarray(source)
    else:
        return cp.asnumpy(source)
