import os
import unittest

import cupy as cp

from sggo.energy import lj
from sggo.local_opt import bfgs
from tests.test_factory import local_opt_test_factory

energy = lj.create()
bfgs_auto = bfgs.create(energy, variant=bfgs.BFGSVariant.AUTO)
bfgs_cpu = bfgs.create(energy, variant=bfgs.BFGSVariant.CPU)
bfgs_gpu = bfgs.create(energy, variant=bfgs.BFGSVariant.GPU)


@unittest.skip("bfgs is currently not reliable")
class TestBFGSAuto(local_opt_test_factory.create(bfgs_auto)):
    pass


@unittest.skip("bfgs is currently not reliable")
class TestBFGSCPU(local_opt_test_factory.create(bfgs_cpu)):
    pass


@unittest.skipIf(not cp.is_available() or "CI" in os.environ, "GPU not available")
@unittest.skip("bfgs is currently not reliable")
class TestBFGSGPU(local_opt_test_factory.create(bfgs_gpu)):
    pass

