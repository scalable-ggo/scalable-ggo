from enum import Enum
from sggo.energy import Energy

from .bfgs_base import BFGS
from .bfgs_impl import BFGSAuto, BFGSCPU, BFGSGPU


class BFGSVariant(Enum):
    AUTO = 1
    CPU = 2
    GPU = 3


def create(
    energy: Energy,
    *,
    variant: BFGSVariant = BFGSVariant.AUTO,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> BFGS:
    kargs = {
        "energy": energy,
        "c1": c1,
        "c2": c2,
    }
    match variant:
        case BFGSVariant.AUTO:
            return BFGSAuto(**kargs)
        case BFGSVariant.CPU:
            return BFGSCPU(**kargs)
        case BFGSVariant.GPU:
            return BFGSGPU(**kargs)
