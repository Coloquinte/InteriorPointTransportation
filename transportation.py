import numpy as np


class TransportationMatrix:
    @staticmethod
    def make_dense_A(N: int, M: int):
        """
        Return the dense constraint matrix
        """
        ret = np.zeros((N + M - 1, N * M))
        # Build source constraints
        for i in range(N):
            for j in range(M):
                ret[i, i * M + j] = 1.0
        # Build sink constraints
        for j in range(M - 1):
            for i in range(N):
                ret[N + j, i * M + j] = 1.0
        return ret

    @staticmethod
    def make_dense_AGAt(N: int, M: int, G):
        """
        Return the dense AGAt matrix obtained with the constraint matrix A and a diagonal matrix G.
        G is passed as a 1D array.
        """
        assert G.ndim == 1
        assert G.size == N * M
        X = G.reshape((N, M))
        Xs = X @ np.ones(M)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = np.ones(N) @ X
        ret = np.zeros((N + M - 1, N + M - 1))
        for i in range(N):
            ret[i, i] = Xs[i]
        for j in range(M - 1):
            ret[N + j, N + j] = Xt[j]
        ret[:N, N:] = X
        ret[N:, :N] = X.transpose()
        return ret

    @staticmethod
    def make_dense_AGAt_explicit(N: int, M: int, G):
        """
        Return the dense AGAt matrix obtained with the constraint matrix A and a diagonal matrix G.
        G is passed as a 1D array.

        The matrix is build explicitly from the result of make_dense_A
        """
        A = TransportationMatrix.make_dense_A(N, M)
        return A @ np.diag(G) @ A.transpose()

    def apply_A(N: int, M: int, t):
        """
        Apply A to the vector t without forming it explicitly
        """
        assert t.ndim == 1
        assert t.size == N * M
        X = t.reshape((N, M))
        Xs = X @ np.ones(M)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = np.ones(N) @ X
        return np.concatenate((Xs, Xt))

    def apply_At(N: int, M: int, t):
        """
        Apply At to the vector t without forming it explicitly
        """
        assert t.ndim == 1
        assert t.size == N + M - 1
        ret = np.zeros((N, M))
        ret[:N, :] = np.expand_dims(t[:N], 1)
        ret[:, :-1] += t[N:]
        return ret.flatten()

    def apply_AGAt(N: int, M: int, G, t):
        """
        Apply AGAt to the vector t without forming it explicitly
        """
        assert G.ndim == 1
        assert G.size == N * M
        assert t.size == N + M - 1
        X = G.reshape((N, M))
        Xs = X @ np.ones(M)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = np.ones(N) @ X
        # Apply as a block matrix
        ts = t[:N]
        tt = t[N:]
        ats = ts * Xs + X @ tt
        att = tt * Xt + X.transpose() @ ts
        return np.concatenate((ats, att))

    def apply_AGAt_inv(N: int, M: int, G, t):
        """
        Apply the inverse of AGAt to the vector t without forming it explicitly, using a block LDLt decomposition.
        Efficient if M is much smaller than N.
        """
        assert G.ndim == 1
        assert G.size == N * M
        assert t.size == N + M - 1
        X = G.reshape((N, M))
        Xs = X @ np.ones(M)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = np.ones(N) @ X

        # Upper diagonal block (just a diagonal matrix)
        Ds = Xs
        # Lower corner block
        L = X.transpose() / Xs
        # Lower diagonal block Dt - L Ds Lt
        Dt = np.diag(Xt) - (L * np.expand_dims(Ds, 0)) @ L.transpose()

        ts = t[:N]
        tt = t[N:]

        # Lower triangular solve
        ats = ts
        att = tt - L @ ts

        # Diagonal solve (requires a solve for the lower block)
        bts = ats / Ds
        btt = np.linalg.solve(Dt, att)

        # Upper triangular solve
        ctt = btt
        cts = bts - L.transpose() @ btt
        ct = np.concatenate((cts, ctt))

        return ct


class TransportationProblem:
    def __init__(self, demands, capacities, costs):
        assert demands.ndim == 1
        assert capacities.ndim == 1
        assert costs.ndim == 2
        self.demands = demands
        self.capacities = capacities
        self.costs = costs
        self.N = demands.size
        self.M = capacities.size
        assert costs.shape == (self.N, self.M)
        assert np.isclose(demands.sum(), capacities.sum())
        assert (demands > 0).all()
        assert (capacities > 0).all()

    @staticmethod
    def make_random(N, M):
        demands = np.random.rand(N) + 1.0e-6
        capacities = np.random.rand(M) + 1.0e-6
        capacities = demands.sum() * capacities / capacities.sum()
        costs = np.random.rand(N, M)
        return TransportationProblem(demands, capacities, costs)

    def initial_solution(self):
        proportion = self.capacities / self.capacities.sum()
        return np.expand_dims(self.demands, 1) * proportion

    def check_solution(self, solution):
        if solution.shape != (self.N, self.M):
            raise RuntimeError("Solution shape is incorrect")
        if not np.isclose(solution.sum(axis=0), self.capacities).all():
            rel_error = (solution.sum(axis=0) - self.capacities) / self.capacities
            rel_error = np.abs(rel_error).max()
            raise RuntimeError(
                f"Capacities do not match (error max {100.0*rel_error:.2f} %)"
            )
        if not np.isclose(solution.sum(axis=1), self.demands).all():
            rel_error = (solution.sum(axis=1) - self.demands) / self.demands
            rel_error = np.abs(rel_error).max()
            raise RuntimeError(
                f"Demands do not match (error max {100.0*rel_error:.2f} %)"
            )

    def initial_dual_solution(self, rel_margin=1.0e-2, abs_margin=1.0e-2):
        ret = np.zeros(self.N + self.M - 1)
        mincost = self.costs.min(axis=1)
        maxcost = self.costs.max(axis=1)
        ret[: self.N] = mincost - rel_margin * (maxcost - mincost) - abs_margin
        return ret

    def check_dual_solution(self, y):
        assert y.shape == (self.N + self.M - 1,)
        Aty = TransportationMatrix.apply_At(self.N, self.M, y).reshape((self.N, self.M))
        if (Aty > self.costs).any():
            raise RuntimeError("Invalid dual solution")

    def value(self, solution):
        assert solution.shape == (self.N, self.M)
        return (solution * self.costs).sum()

    def apply_A(self, t):
        return TransportationMatrix.apply_A(self.N, self.M, t)

    def apply_At(self, t):
        return TransportationMatrix.apply_At(self.N, self.M, t)

    def apply_AGAt(self, G, t):
        return TransportationMatrix.apply_AGAt(self.N, self.M, G, t)

    def apply_AGAt_inv(self, G, t):
        return TransportationMatrix.apply_AGAt_inv(self.N, self.M, G, t)

    def solve_affine_scaling(self, beta=0.5, epsilon=1.0e-6, residual_tol=1.0e-6):
        x = self.initial_solution().flatten()
        c = self.costs.flatten()
        k = 0
        while True:
            # Dual variable computation: (A D^2 At)^-1 A D^2 c
            x2 = x * x
            y = x2 * c
            y = self.apply_A(y)
            y = self.apply_AGAt_inv(x2, y)
            # Reduced cost computation
            r = c - self.apply_At(y)
            if (r >= -residual_tol).all() and np.dot(x, r) < epsilon:
                return x.reshape((self.N, self.M))
            x = x - beta * x2 * r / np.linalg.norm(x * r)
            k += 1
