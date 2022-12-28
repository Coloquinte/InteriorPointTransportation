import numpy as np
import unittest

from interior_point_transportation import TransportationProblem

params_list = [(4, 5), (5, 3), (5, 8)]


class MatrixTest(unittest.TestCase):
    def test_dense_matrix(self):
        """
        Test the dense AGAt against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                A = TransportationProblem.make_dense_A(N, M)
                G = np.random.rand(N * M)
                AGAt = TransportationProblem.make_dense_AGAt(N, M, G)
                expected = A @ np.diag(G) @ A.transpose()
                self.assertTrue(np.isclose(AGAt, expected).all())

    def test_apply_A(self):
        """
        Test the implicit A against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                A = TransportationProblem.make_dense_A(N, M)
                x = np.random.rand(N * M)
                Ax = TransportationProblem.apply_A(N, M, x)
                expected = A @ x
                self.assertTrue(np.isclose(Ax, expected).all())

    def test_apply_AGAt(self):
        """
        Test the implicit AGAt against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                G = np.random.rand(N * M)
                x = np.random.rand(N + M - 1)
                AGAtx = TransportationProblem.apply_AGAt(N, M, G, x)
                AGAt = TransportationProblem.make_dense_AGAt(N, M, G)
                expected = AGAt @ x
                self.assertTrue(np.isclose(AGAtx, expected).all())

    def test_apply_AGAt_inv(self):
        """
        Test the implicit AGAt^-1 against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                G = 0.01 + np.random.rand(N * M)
                x = np.random.rand(N + M - 1)
                AGAt_inv_x = TransportationProblem.apply_AGAt_inv(N, M, G, x)
                AGAt = TransportationProblem.make_dense_AGAt(N, M, G)
                expected = np.linalg.solve(AGAt, x)
                self.assertTrue(np.isclose(AGAt_inv_x, expected).all())


if __name__ == "__main__":
    unittest.main()
