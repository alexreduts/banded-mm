"""
Unittests for the currently under development functions

"""

from unittest import TestCase

import numpy as np

from banded_mm.xGBMM_naive_copy import xGBMM_naive_copy
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

        C_rect = xGBMM_naive(A_rect, 0, 0, B_rect, 0, 0, 3, 2)
        print("\n", C_rect-REF_rect)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Diagonal Case failed"
        )

        C_square = xGBMM_streamed(A_square, 0, 0, B_square, 0, 0, 3, 2)
        print("\n", C_square-REF_square)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Diagonal Case failed"
        )

    def test_banded(self):
        A_rect = banded_matrix_generator(5, 10, 2, 3)
        B_rect = banded_matrix_generator(10, 5, 2, 3)

        A_square = banded_matrix_generator(10, 10, 2, 3)
        B_square = banded_matrix_generator(10, 10, 2, 3)

        REF_rect = A_rect @ B_rect
        REF_square = A_square @ B_square

        C_rect = xGBMM_streamed(A_rect, 2, 3, B_rect, 2, 3, 3, 2)
        print("\n", C_rect-REF_rect)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Diagonal Case failed"
        )

        C_square = xGBMM_streamed(A_square, 2, 3, B_square, 2, 3, 3, 2)
        print("\n", C_square-REF_square)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Diagonal Case failed"
        )

    def test_inner_loop_slicing(self):

        print(_slicer(1, 5, 7, 0, 2, 2))


    def test_inner_loop_banded_no_overlap(self):
        A_rect = banded_matrix_generator(5, 10, 1, 2)
        B_rect = banded_matrix_generator(10, 5, 1, 2)

        A_square = banded_matrix_generator(10, 10, 2, 3)
        B_square = banded_matrix_generator(10, 10, 2, 3)

        REF_rect = A_rect @ B_rect
        REF_square = A_square @ B_square

        C_rect = xGBMM_streamed(A_rect, 1, 2, B_rect, 1, 2, 5, 2)
        print("\n", C_rect-REF_rect)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Inner Loop Banded NO Overlap failed"
        )

        C_square = xGBMM_streamed(A_square, 2, 3, B_square, 2, 3, 10, 2)
        print("\n", C_square-REF_square)
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

        C_rect = xGBMM_streamed(A_rect, 2, 3, B_rect, 2, 3, 5, 2)
        print("\n", C_rect-REF_rect)
        self.assertTrue(
            np.allclose(C_rect, REF_rect),
            "Rectangular Inner Loop Banded With Overlap failed"
        )

        C_square = xGBMM_streamed(A_square, 2, 3, B_square, 2, 3, 10, 2)
        print("\n", C_square-REF_square)
        self.assertTrue(
            np.allclose(C_square, REF_square),
            "Squared Inner Loop Banded With Overlap failed"
        )