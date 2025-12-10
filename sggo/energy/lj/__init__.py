from enum import Enum
from .lj_base import LJ

from .lj_impl import LJAuto, LJCPU, LJGPU
from .lj_impl_kernel import LJGPUKernel
from .lj_threaded import LJThreaded


class LJVariant(Enum):
    AUTO = 1
    CPU = 2
    CPUThreaded = 3
    GPU = 4
    GPUKERNEL = 5


def create(variant: LJVariant = LJVariant.AUTO) -> LJ:
    match variant:
        case LJVariant.AUTO:
            return LJAuto()
        case LJVariant.CPU:
            return LJCPU()
        case LJVariant.CPUThreaded:
            return LJThreaded()
        case LJVariant.GPU:
            return LJGPU()
        case LJVariant.GPUKERNEL:
            return LJGPUKernel()
