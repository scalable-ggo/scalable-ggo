from enum import Enum
from sggo.energy import Energy

from .cg_base import CG
from .cg_impl import CGAuto, CGCPU, CGGPU


class CGVariant(Enum):
    AUTO = 1
    CPU = 2
    GPU = 3


def create(
    energy: Energy,
    *,
    variant: CGVariant = CGVariant.AUTO,
    c1: float = 1e-4,
    c2: float = 0.4,
) -> CG:
    kargs = {
        "energy": energy,
        "c1": c1,
        "c2": c2,
    }
    match variant:
        case CGVariant.AUTO:
            return CGAuto(**kargs)
        case CGVariant.CPU:
            return CGCPU(**kargs)
        case CGVariant.GPU:
            return CGGPU(**kargs)
