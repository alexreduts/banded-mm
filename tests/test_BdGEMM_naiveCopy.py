"""
Unittests for the currently under development functions

"""

from unittest import TestCase

import numpy as np

from banded_mm.BdGEMM_naiveCopy import BdGEMM_naiveCopy
from banded_mm.matrix_utils import banded_matrix_generator

class TestxGBMM(TestCase):

    def setUP(self):
        pass

    def test_diagonal(self):
        A_rect = banded_matrix_generator(5, 10, 0, 0)
        B_rect = banded_matrix_generator(10, 5, 0, 0)

        A_square = banded_matrix_generator(10, 10, 0, 0)
        B_square = banded_matrix_generator(10, 10, 0, 0)

        REF_rect = A_rect @ B_rect
        REF_square = A_square @ B_square

        C_rect = BdGEMM_naiveCopy(A_rect, 0, 0, B_rect, 0, 0, 3, 2)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Diagonal Case failed"
        )

        C_square = BdGEMM_naiveCopy(A_square, 0, 0, B_square, 0, 0, 3, 2)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Diagonal Case failed"
        )

    def test_dense(self):
        A_rect = banded_matrix_generator(5, 10, 4, 9)
        B_rect = banded_matrix_generator(10, 5, 9, 4)

        A_square = banded_matrix_generator(10, 10, 9, 9)
        B_square = banded_matrix_generator(10, 10, 9, 9)

        REF_rect = A_rect @ B_rect
        REF_square = A_square @ B_square

        C_rect = BdGEMM_naiveCopy(A_rect, 4, 9, B_rect, 9, 4, 3, 2)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Dense Case failed"
        )

        C_square = BdGEMM_naiveCopy(A_square, 9, 9, B_square, 9, 9, 3, 2)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Dense Case failed"
        )

    def test_banded(self):
        A_rect = banded_matrix_generator(5, 10, 2, 3)
        B_rect = banded_matrix_generator(10, 5, 2, 3)

        A_square = banded_matrix_generator(10, 10, 2, 3)
        B_square = banded_matrix_generator(10, 10, 2, 3)

        REF_rect = A_rect @ B_rect
        REF_square = A_square @ B_square

        C_rect = BdGEMM_naiveCopy(A_rect, 2, 3, B_rect, 2, 3, 3, 2)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Banded Case failed"
        )

        C_square = BdGEMM_naiveCopy(A_square, 2, 3, B_square, 2, 3, 3, 2)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Banded Case failed"
        )

    def test_inner_loop_banded_no_overlap(self):
        A_rect = banded_matrix_generator(5, 10, 1, 2)
        B_rect = banded_matrix_generator(10, 5, 1, 2)

        A_square = banded_matrix_generator(10, 10, 2, 3)
        B_square = banded_matrix_generator(10, 10, 2, 3)

        REF_rect = A_rect @ B_rect
        REF_square = A_square @ B_square

        C_rect = BdGEMM_naiveCopy(A_rect, 1, 2, B_rect, 1, 2, 5, 2)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Inner Loop Banded NO Overlap failed"
        )

        C_square = BdGEMM_naiveCopy(A_square, 2, 3, B_square, 2, 3, 10, 2)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Inner Loop Banded NO Overlap failed"
        )

    def test_inner_loop_banded_with_overlap(self):
        A_rect = banded_matrix_generator(5, 10, 2, 3)
        B_rect = banded_matrix_generator(10, 5, 2, 3)

        A_square = banded_matrix_generator(10, 10, 4, 5)
        B_square = banded_matrix_generator(10, 10, 4, 5)

        REF_rect = A_rect @ B_rect
        REF_square = A_square @ B_square

        C_rect = BdGEMM_naiveCopy(A_rect, 2, 3, B_rect, 2, 3, 5, 3)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Inner Loop Banded With Overlap failed"
        )

        C_square = BdGEMM_naiveCopy(A_square, 4, 5, B_square, 4, 5, 10, 3)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Inner Loop Banded With Overlap failed"
        )