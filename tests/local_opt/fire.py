

from sggo.energy import lj
from sggo.local_opt import fire
from tests import utils
from tests.test_factory import local_opt_test_factory

energy = lj.create()
fire_auto = fire.create(energy, variant=fire.FIREVariant.AUTO)
fire_cpu = fire.create(energy, variant=fire.FIREVariant.CPU)
fire_gpu = fire.create(energy, variant=fire.FIREVariant.GPU)


class TestFIREAuto(local_opt_test_factory.create(fire_auto)):
    pass


class TestFIRECPU(local_opt_test_factory.create(fire_cpu)):
    pass


@utils.skip_in_ci
class TestFIREGPU(local_opt_test_factory.create(fire_gpu)):
    pass

