from enum import Enum

from .lj_base import LJ
from .lj_impl import LJCPU, LJGPU, LJAuto
from .lj_impl_kernel import LJGPUKernel


class LJVariant(Enum):
    AUTO = 1
    CPU = 2
    GPU = 3
    GPUKERNEL = 4


def create(variant: LJVariant = LJVariant.AUTO) -> LJ:
    match variant:
        case LJVariant.AUTO:
            return LJAuto()
        case LJVariant.CPU:
            return LJCPU()
        case LJVariant.GPU:
            return LJGPU()
        case LJVariant.GPUKERNEL:
            return LJGPUKernel()
