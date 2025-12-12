import numpy as np
from sggo.types import NDArray


def foo(x: NDArray) -> NDArray:
    return x + np.asarray(42)
