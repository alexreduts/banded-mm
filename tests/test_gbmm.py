"""
Unittests for the currently under development functions

"""

from unittest import TestCase

import numpy as np

from banded_mm.gbmm import gbmm_BLK, gbmm_BB
from banded_mm.matrix_utils import banded_matrix_generator

class TestGBMM(TestCase):

    def setUP(self):
        pass

    def test_gbmm_BLK(self):
        A = banded_matrix_generator(8, 2, 2)
        B = banded_matrix_generator(8, 2, 2)
        C_ref = np.matmul(A,B)
        C = np.zeros((A.shape[0], B.shape[1]))
        C = gbmm_BLK(C, A, B, 2, 2)
        self.assertTrue(np.allclose(C, C_ref))

    def test_gbmm_BB(self):
        A = banded_matrix_generator(30, 1, 7)
        B = banded_matrix_generator(30, 2, 0)
        C_ref = np.matmul(A,B)
        C = np.zeros((A.shape[0], B.shape[1]))
        C = gbmm_BB(C, A, 1, 7, B, 2, 0)
        self.assertTrue(np.allclose(C, C_ref))