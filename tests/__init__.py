from .test_example import TestExample
from .global_opt.test_genetic import TestGenetic
from .energy.lj import TestLJAuto, TestLJCPU, TestLJGPU, TestLJGPUKernel
from .local_opt.bfgs import TestBFGSAuto, TestBFGSCPU, TestBFGSGPU
from .local_opt.cg import TestCGAuto, TestCGCPU, TestCGGPU
from .local_opt.fire import TestFIREAuto, TestFIRECPU, TestFIREGPU

__all__ = [
    "TestExample",
    "TestGenetic",

    "TestLJAuto",
    "TestLJCPU",
    "TestLJGPU",
    "TestLJGPUKernel",

    "TestBFGSAuto",
    "TestBFGSCPU",
    "TestBFGSGPU",

    "TestCGAuto",
    "TestCGCPU",
    "TestCGGPU",

    "TestFIREAuto",
    "TestFIRECPU",
    "TestFIREGPU",
]
