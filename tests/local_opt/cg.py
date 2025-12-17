

from sggo.energy import lj
from sggo.local_opt import cg
from tests import utils
from tests.test_factory import local_opt_test_factory

energy = lj.create()
cg_auto = cg.create(energy, variant=cg.CGVariant.AUTO)
cg_cpu = cg.create(energy, variant=cg.CGVariant.CPU)
cg_gpu = cg.create(energy, variant=cg.CGVariant.GPU)


class TestCGAuto(local_opt_test_factory.create(cg_auto)):
    pass


class TestCGCPU(local_opt_test_factory.create(cg_cpu)):
    pass


@utils.skip_in_ci
class TestCGGPU(local_opt_test_factory.create(cg_gpu)):
    pass

