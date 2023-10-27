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

        C = np.zeros((self.A.shape[0], self.B.shape[1]))
        C = naive_blocked_banded_mm(C, self.A, self.B, 2, 2)
        self.assertTrue(np.allclose(C, self.C_ref))