import numpy as np
import unittest

from transportation import TransportationMatrix, TransportationProblem

params_list = [(4, 5), (5, 3), (5, 8)]


class TransportationTest(unittest.TestCase):
    def test_dense_matrix(self):
        """
        Test the dense AGAt against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                A = TransportationMatrix.make_dense_A(N, M)
                G = np.random.rand(N * M)
                AGAt = TransportationMatrix.make_dense_AGAt(N, M, G)
                expected = A @ np.diag(G) @ A.transpose()
                self.assertTrue(np.isclose(AGAt, expected).all())

    def test_apply_A(self):
        """
        Test the implicit A against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                A = TransportationMatrix.make_dense_A(N, M)
                x = np.random.rand(N * M)
                Ax = TransportationMatrix.apply_A(N, M, x)
                expected = A @ x
                self.assertTrue(np.isclose(Ax, expected).all())

    def test_apply_At(self):
        """
        Test the implicit At against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                A = TransportationMatrix.make_dense_A(N, M)
                x = np.random.rand(N + M - 1)
                Atx = TransportationMatrix.apply_At(N, M, x)
                expected = A.transpose() @ x
                self.assertTrue(np.isclose(Atx, expected).all())

    def test_apply_AGAt(self):
        """
        Test the implicit AGAt against the explicit result
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                G = np.random.rand(N * M)
                x = np.random.rand(N + M - 1)
                AGAtx = TransportationMatrix.apply_AGAt(N, M, G, x)
                AGAt = TransportationMatrix.make_dense_AGAt(N, M, G)
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
                AGAt_inv_x = TransportationMatrix.apply_AGAt_inv(N, M, G, x)
                AGAt = TransportationMatrix.make_dense_AGAt(N, M, G)
                expected = np.linalg.solve(AGAt, x)
                self.assertTrue(np.isclose(AGAt_inv_x, expected).all())

    def test_initial_primal_solution(self):
        """
        Test that the initial solution from the transportation problem is correct
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                prob = TransportationProblem.make_random(N, M)
                sol = prob.initial_solution()
                prob.check_solution(sol)

    def test_initial_dual_solution(self):
        """
        Test that the initial dual solution from the transportation problem is correct
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                prob = TransportationProblem.make_random(N, M)
                sol = prob.initial_dual_solution()
                prob.check_dual_solution(sol)

    def test_affine_scaling(self):
        """
        Run the primal affine scaling algorithm
        """
        for N, M in params_list:
            with self.subTest(N=N, M=M):
                prob = TransportationProblem.make_random(N, M)
                sol = prob.solve_affine_scaling()
                prob.check_solution(sol)


if __name__ == "__main__":
    unittest.main()
