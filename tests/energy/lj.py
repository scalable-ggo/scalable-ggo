

from sggo.energy import lj
from tests import utils
from tests.test_factory import lj_test_factory

lj_auto = lj.create(variant=lj.LJVariant.AUTO)
lj_cpu = lj.create(variant=lj.LJVariant.CPU)
lj_gpu = lj.create(variant=lj.LJVariant.GPU)
lj_gpukernel = lj.create(variant=lj.LJVariant.GPUKERNEL)


class TestLJAuto(lj_test_factory.create(lj_auto)):
    pass


class TestLJCPU(lj_test_factory.create(lj_cpu)):
    pass


@utils.skip_in_ci
class TestLJGPU(lj_test_factory.create(lj_gpu)):
    pass


@utils.skip_in_ci
class TestLJGPUKernel(lj_test_factory.create(lj_gpukernel)):
    pass
