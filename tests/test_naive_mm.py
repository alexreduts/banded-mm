"""
Unittests for the currently under development functions

"""

from unittest import TestCase

import numpy as np

from banded_mm.naive_mm import naive_blocked_banded_mm
from banded_mm.matrix_utils import banded_matrix_generator

class TestGBMM(TestCase):

    def setUP(self):
        pass

    def test_gbmm_BLK(self):
        self.A = banded_matrix_generator(8, 2, 2)
        self.B = banded_matrix_generator(8, 2, 2)
        self.C_ref = np.matmul(self.A,self.B)

        C = naive_blocked_banded_mm(self.A, 5, self.B, 5, 3)
        self.assertTrue(np.allclose(C, self.C_ref))