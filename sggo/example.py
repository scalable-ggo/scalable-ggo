import numpy as np


def foo(x: np.typing.ArrayLike) -> np.typing.ArrayLike:
    return x + np.asarray(42)
