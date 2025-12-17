import unittest

import cupy as cp

from sggo.energy import lj
from tests.test_factory import lj_test_factory

lj_auto = lj.create(variant=lj.LJVariant.AUTO)
lj_cpu = lj.create(variant=lj.LJVariant.CPU)
lj_gpu = lj.create(variant=lj.LJVariant.GPU)
lj_gpukernel = lj.create(variant=lj.LJVariant.GPUKERNEL)


class TestLJAuto(lj_test_factory.create(lj_auto)):
    pass


class TestLJCPU(lj_test_factory.create(lj_cpu)):
    pass


@unittest.skipIf(not cp.is_available(), "GPU not available")
class TestLJGPU(lj_test_factory.create(lj_gpu)):
    pass


@unittest.skipIf(not cp.is_available(), "GPU not available")
class TestLJGPUKernel(lj_test_factory.create(lj_gpukernel)):
    pass
