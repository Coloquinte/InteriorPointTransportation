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
        self.demands, self.capacities = TransportationProblem.normalize_demands_capas(
            demands, capacities
        )
        self.costs = TransportationProblem.normalize_costs(costs)
        self.N = demands.size
        self.M = capacities.size

    @staticmethod
    def normalize_costs(costs):
        assert costs.ndim == 2
        costs = costs - costs.min(axis=1, keepdims=True)
        costs = costs / costs.mean()
        return costs

    def normalize_demands_capas(demands, capacities):
        assert demands.ndim == 1
        assert capacities.ndim == 1
        assert (demands > 0).all()
        assert (capacities > 0).all()
        assert np.isclose(demands.sum(), capacities.sum())
        total_dem = demands.sum()
        return demands / total_dem, capacities / total_dem

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

    def initial_dual_solution(self, margin=1.0e-2):
        assert (self.costs >= 0.0).all()
        assert np.isclose(self.costs.mean(), 1.0)
        ret = np.zeros(self.N + self.M - 1)
        mincost = self.costs.min(axis=1)
        maxcost = self.costs.max(axis=1)
        ret[: self.N] = -margin / 2
        ret[self.N :] = -margin / 2
        return ret

    def check_dual_solution(self, y):
        assert y.shape == (self.N + self.M - 1,)
        Aty = self.apply_At(y).reshape((self.N, self.M))
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

    def solve_affine_scaling(self, beta=0.995, epsilon=1.0e-6, residual_tol=1.0e-6):
        x = self.initial_solution().flatten()
        c = self.costs.flatten()
        k = 0
        print(f"Initial value {self.value(x.reshape((self.N, self.M)))}")
        while True:
            # Dual variable computation: (A D^2 At)^-1 A D^2 c
            x2 = x * x
            y = x2 * c
            y = self.apply_A(y)
            y = self.apply_AGAt_inv(x2, y)
            # Reduced cost computation
            s = c - self.apply_At(y)
            residual = np.dot(x, s)
            min_reduced = s.min()
            value = self.value(x.reshape((self.N, self.M)))
            print(
                f"Iter {k+1}: value {value}, residual {residual}, min reduced cost {min_reduced}"
            )
            if min_reduced >= -residual_tol and residual < epsilon:
                return x.reshape((self.N, self.M))
            x = x - beta * x2 * s / np.linalg.norm(x * s)
            k += 1

    def solve_dual_affine_scaling(
        self, beta=0.995, epsilon=1.0e-6, primal_tol=1.0e-2
    ):
        y = self.initial_dual_solution().flatten()
        c = self.costs.flatten()
        b = np.concatenate((self.demands, self.capacities))[:-1]
        k = 0
        while True:
            s = c - self.apply_At(y)
            assert (s >= 0).all()
            s_msquare = 1.0 / s**2
            d_y = self.apply_AGAt_inv(s_msquare, b)
            d_s = -self.apply_At(d_y)
            alpha = -beta / min(min(d_s / s), -1)
            y = y + alpha * d_y
            x = -s_msquare * d_s
            residual = np.dot(x, s)
            min_primal = x.min()
            value = self.value(x.reshape((self.N, self.M)))
            print(
                f"Iter {k+1}: value {value}, residual {residual}, min primal value {min_primal}"
            )
            if min_primal >= -primal_tol and residual < epsilon:
                return x.reshape((self.N, self.M))
            k = k + 1

    def solve_highs(self):
        import highspy

        solver = highspy.Highs()
        inf = highspy.kHighsInf
        min_vars = np.zeros(self.N * self.M)
        max_vars = np.full(self.N * self.M, inf)
        solver.addVars(self.N * self.M, min_vars, max_vars)
        for i in range(self.N):
            row_inds = np.zeros(self.M, dtype=np.int32)
            row_vals = np.ones(self.M, dtype=np.float64)
            for j in range(self.M):
                row_inds[j] = self.M * i + j
            solver.addRow(self.demands[i], self.demands[i], self.M, row_inds, row_vals)
        for j in range(self.M):
            row_inds = np.zeros(self.N, dtype=np.int32)
            row_vals = np.ones(self.N, dtype=np.float64)
            for i in range(self.N):
                row_inds[i] = self.M * i + j
            solver.addRow(
                self.capacities[j], self.capacities[j], self.N, row_inds, row_vals
            )
        col_inds = np.zeros(self.N * self.M, dtype=np.int32)
        col_costs = self.costs.flatten()
        for i in range(self.N * self.M):
            col_inds[i] = i
        solver.changeColsCost(self.N * self.M, col_inds, col_costs)
        options = solver.getOptions()
        options.presolve = "off"
        options.solver = "ipm"
        options.run_crossover = "off"
        options.log_to_console = True
        solver.passOptions(options)
        solver.run()
        solution = solver.getSolution()
        return np.asarray(solution.col_value, dtype=np.float64).reshape(self.N, self.M)

    def solve_ortools(self):
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("GLOP")
        x = [
            [solver.NumVar(0, solver.infinity(), f"x_{i}_{j}") for j in range(self.M)]
            for i in range(self.N)
        ]
        for i in range(self.N):
            solver.Add(solver.Sum(x[i]) == self.demands[i])
        for j in range(self.M):
            solver.Add(
                solver.Sum([x[i][j] for i in range(self.N)]) == self.capacities[j]
            )
        solver.Minimize(
            solver.Sum(
                [
                    self.costs[i, j] * x[i][j]
                    for i in range(self.N)
                    for j in range(self.M)
                ]
            )
        )

        solver.EnableOutput()
        status = solver.Solve()
        assert status == pywraplp.Solver.OPTIMAL
        return np.asarray(
            [[x[i][j].solution_value() for j in range(self.M)] for i in range(self.N)]
        )
