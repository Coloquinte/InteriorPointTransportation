import numpy as np
import unittest

from interior_point_transportation import TransportationProblem

params_list = [(4, 2), (2, 2), (5, 8)]


class MatrixTest(unittest.TestCase):
    def test_dense_matrix(self):
        for N, M in params_list:
            A = TransportationProblem.make_dense_A(N, M)
            G = np.random.rand(N * M)
            AGAt = TransportationProblem.make_dense_AGAt(N, M, G)
            expected = A @ np.diag(G) @ A.transpose()
            self.assertTrue(np.isclose(AGAt, expected).all())

    def test_apply_A(self):
        for N, M in params_list:
            A = TransportationProblem.make_dense_A(N, M)
            x = np.random.rand(N * M)
            Ax = TransportationProblem.apply_A(N, M, x)
            expected = A @ x
            self.assertTrue(np.isclose(Ax, expected).all())

    def test_apply_AGAt(self):
        for N, M in params_list:
            G = np.random.rand(N * M)
            x = np.random.rand(N + M - 1)
            AGAtx = TransportationProblem.apply_AGAt(N, M, G, x)
            AGAt = TransportationProblem.make_dense_AGAt(N, M, G)
            expected = AGAt @ x
            self.assertTrue(np.isclose(AGAtx, expected).all())
