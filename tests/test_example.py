import unittest

import numpy as np
from sggo.example import foo


class TestExample(unittest.TestCase):
    def test_foo_scalar(self):
        self.assertEqual(foo(10), 52)

    def test_foo_list(self):
        np.testing.assert_equal(foo([1, 2, 3]), [43, 44, 45])
