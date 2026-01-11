from enum import Enum
from sggo.energy import Energy

from .fire_base import FIRE
from .fire_impl import FIREAuto, FIRECPU, FIREGPU
from .fire_impl_kernel import FIREGPUKernel


class FIREVariant(Enum):
    AUTO = 1
    CPU = 2
    GPU = 3
    GPUKERNEL = 4


def create(
    energy: Energy,
    *,
    variant: FIREVariant = FIREVariant.AUTO,
    maxstep: float = 0.2,
    dt_start: float = 0.1,
    dtmax: float = 1.0,
    nmin: int = 5,
    finc: float = 1.1,
    fdec: float = 0.5,
    a_start: float = 0.1,
    fa: float = 0.99,
) -> FIRE:
    kargs = {
        "energy": energy,
        "maxstep": maxstep,
        "dt_start": dt_start,
        "dtmax": dtmax,
        "nmin": nmin,
        "finc": finc,
        "fdec": fdec,
        "a_start": a_start,
        "fa": fa,
    }
    match variant:
        case FIREVariant.AUTO:
            return FIREAuto(**kargs)
        case FIREVariant.CPU:
            return FIRECPU(**kargs)
        case FIREVariant.GPU:
            return FIREGPU(**kargs)
        case FIREVariant.GPUKERNEL:
            return FIREGPUKernel(**kargs)
