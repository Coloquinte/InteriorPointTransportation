import numpy as np

from numpy.linalg import norm

from scipy import sparse
from scipy.sparse.linalg import spsolve


def initial_point(a, b, c, use_umfpack=False):

    n = a.shape[1]
    e = np.ones((n,))

    # solution for min norm(s) s.t. A'*y + s = c
    # y =sparse.linalg.cg(a*a.T, a*c,tol=1e-7)[0]
    y = spsolve(a * a.T, a * c, use_umfpack=use_umfpack)

    # y2 =sparse.linalg.cgs(A*A.T, A*c)[0]
    # y2 =sparse.linalg.gmres(A*A.T, A*c,)[0]

    s = c - a.T * y

    # solution for min norm(x) s.t. Ax = b
    x = a.T * spsolve(a * a.T, b, use_umfpack=use_umfpack)
    # print(c.T.dot(x))
    # x = a.T*sparse.linalg.cg(a*a.T, b,tol=1e-7)[0]

    # delta_x and delta_s
    delta_x = max(-1.5 * np.min(x), 0)
    delta_s = max(-1.5 * np.min(s), 0)

    # delta_x_c and delta_s_c
    pdct = 0.5 * (x + delta_x * e).dot(s + delta_s * e)
    delta_x_c = delta_x + pdct / (np.sum(s) + n * delta_s)
    delta_s_c = delta_s + pdct / (np.sum(x) + n * delta_x)

    print(
        f"delta_x={delta_x}\ndelta_s={delta_s}\ndelta_x_c={delta_x_c}\ndelta_s_c={delta_s_c}\n"
    )
    # output
    x0 = x + delta_x_c * e
    s0 = s + delta_s_c * e
    y0 = y
    return x0, y0, s0


def newton_direction(r_b, r_c, r_x_s, a, m, n, x, s, lu, error_check=0, use_lu=True):

    p1 = -r_b
    p2 = -r_c + r_x_s / x
    rhs = np.hstack((-r_b, -r_c + r_x_s / x))
    d_2 = -np.minimum(1e16, s / x)
    b = sparse.vstack(
        (
            sparse.hstack((sparse.coo_matrix((m, m)), a)),
            sparse.hstack((a.T, sparse.diags([d_2], [0]))),
        )
    )

    # dy       0    A           p1
    #     =                x     
    # dx       At   d_2         p2


    # dx       d_2    At        p2
    #     =                x     
    # dy       A      0         p1

    # Block cholesky
    u, v = p2, p1

    v = v - a @ (u / d_2)

    u = u / d_2
    v = np.linalg.solve(-a @ np.diag(1.0/d_2) @ a.T, v)

    u = u - (a.T * v) / d_2

    # ldl' factorization
    # if L and D are not provided, we calc new factorization; otherwise,
    # reuse them

    if use_lu:
        if lu is None:
            lu = sparse.linalg.splu(b.tocsc())
            # wikipedia says it uses Mehrotra cholesky but the matrix i'm getting is not definite positive
            # scikits.sparse.cholmod.cholesky fails without a warning

        sol = lu.solve(rhs)
    else:
        sol = sparse.linalg.cg(b, rhs, tol=1e-5)[0]
        # assert(np.max(np.abs(B*sol-rhs))<1e-5)


    dy = sol[:m]
    dx = sol[m : m + n]
    ds = -(r_x_s + s * dx) / x

    AGAt = a @ sparse.diags(d_2) @ a.T
    AGAt_inv = sparse.linalg.inv(AGAt)

    import pdb; pdb.set_trace()

    if error_check == 1:
        print(
            "error = %6.2e"
            % (
                norm(a.T * dy + ds + r_c)
                + norm(a * dx + r_b)
                + norm(s * dx + x * ds + r_x_s)
            )
        )
        print("\t + err_d = %6.2e" % (norm(a.T * dy + ds + r_c)))
        print("\t + err_p = %6.2e" % (norm(a * dx + r_b)))
        print("\t + err_gap = %6.2e\n" % (norm(s * dx + x * ds + r_x_s)))

    return dx, dy, ds, lu


def step_size(x, s, d_x, d_s, eta=0.9995):
    alpha_x = -1 / min(min(d_x / x), -1)
    alpha_x = min(1, eta * alpha_x)
    alpha_s = -1 / min(min(d_s / s), -1)
    alpha_s = min(1, eta * alpha_s)
    return alpha_x, alpha_s


def mpc_sol(
    a,
    b,
    c,
    max_iter=100,
    eps=1e-9,
    theta=0.9995,
    verbose=2,
    error_check=False,
    callback=None,
    x = None,
    y = None,
    s = None
):

    a = sparse.coo_matrix(a)
    c = np.squeeze(np.array(c))
    b = np.squeeze(np.array(b))

    # Initialization

    m, n = a.shape
    alpha_x = 0
    alpha_s = 0

    if verbose > 1:
        print(
            "\n%3s %6s %9s %11s %9s %9s %9s\n"
            % ("ITER", "COST", "MU", "RESIDUAL", "ALPHAX", "ALPHAS", "MAXVIOL")
        )

    # Choose initial point
    if x is None or y is None or s is None:
        x, y, s = initial_point(a, b, c)

    bc = 1 + max([norm(b), norm(c)])

    # Start the loop
    niter_done = 0

    for niter in range(max_iter):
        # Compute residuals and update mu
        r_b = a * x - b
        r_c = a.T * y + s - c
        r_x_s = x * s
        mu = np.mean(r_x_s)
        f = c.T.dot(x)

        # Check relative decrease in residual, for purposes of convergence test
        residual = norm(np.hstack((r_b, r_c, r_x_s)) / bc)

        if verbose > 1:
            maxviol = max(np.max(np.abs(r_b)), np.max(-x))
            print(
                "%3d %9.2e %9.2e %9.2e %9.4g %9.4g %9.2e"
                % (niter, f, mu, residual, alpha_x, alpha_s, maxviol)
            )

        if callback is not None:
            callback(x, niter)

        if residual < eps:
            break

        # ----- Predictor step -----

        # Get affine-scaling direction
        dx_aff, dy_aff, ds_aff, lu = newton_direction(
            r_b, r_c, r_x_s, a, m, n, x, s, None, error_check
        )

        # Get affine-scaling step length
        alpha_x_aff, alpha_s_aff = step_size(x, s, dx_aff, ds_aff, 1)
        mu_aff = (x + alpha_x_aff * dx_aff).dot(s + alpha_s_aff * ds_aff) / n

        # Set central parameter
        sigma = (mu_aff / mu) ** 3

        # ----- Corrector step -----

        # Set up right hand sides
        r_x_s = r_x_s + dx_aff * ds_aff - sigma * mu * np.ones((n))

        # Get corrector's direction
        dx_cc, dy_cc, ds_cc, lu = newton_direction(
            r_b, r_c, r_x_s, a, m, n, x, s, lu, error_check
        )

        # Compute search direction and step
        dx = dx_aff + dx_cc
        dy = dy_aff + dy_cc
        ds = ds_aff + ds_cc

        alpha_x, alpha_s = step_size(x, s, dx, ds, theta)

        # Update iterates
        x = x + alpha_x * dx
        y = y + alpha_s * dy
        s = s + alpha_s * ds

        if niter == max_iter and verbose > 1:
            print("max_iter reached!\n")
        niter_done = niter

    if verbose > 0:
        print("\nDONE! [m,n] = [%d, %d], N = %d\n" % (m, n, niter))

    f = c.T.dot(x)

    return f, x, y, s, niter_done



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
        mean_dem = demands.mean()
        return demands / mean_dem, capacities / mean_dem

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

    def solve_affine_scaling(self, beta=0.999, epsilon=1.0e-6, residual_tol=1.0e-6):
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

    def solve_dual_affine_scaling(self, beta=0.99, epsilon=1.0e-6, primal_tol=1.0e-2):
        y = self.initial_dual_solution().flatten()
        c = self.costs.flatten()
        b = np.concatenate((self.demands, self.capacities))[:-1]
        k = 0
        while True:
            s = c - self.apply_At(y)
            assert (s >= 0).all()
            s_msquare = 1.0 / (s + 1.0e-10)**2
            d_y = self.apply_AGAt_inv(s_msquare, b)
            d_s = -self.apply_At(d_y)
            alpha = -beta / min(d_s / s)
            y = y + alpha * d_y
            x = -s_msquare * d_s
            residual = np.dot(x, s)
            min_primal = x.min()
            value = self.value(x.reshape((self.N, self.M)))
            print(
                f"Iter {k+1}: value {value}, residual {residual}, min primal value {min_primal}, step {alpha}"
            )
            if min_primal >= -primal_tol and residual < epsilon:
                return x.reshape((self.N, self.M))
            k = k + 1

    def solve_mpc(self):
        c = self.costs.flatten()
        b = np.concatenate((self.demands, self.capacities))[:-1]
        a = TransportationMatrix.make_dense_A(self.N, self.M)

        x = self.initial_solution().flatten()
        y = self.initial_dual_solution(1.0).flatten()
        s = c - self.apply_At(y)
        x = mpc_sol(a, b, c, x=x, y=y, s=s)[1]
        #x = mpc_sol(a, b, c)[1]
        return x.reshape((self.N, self.M))

    def solve_primal_dual(self, beta=0.995, epsilon=1.0e-6):
        x = self.initial_solution().flatten()
        y = self.initial_dual_solution().flatten()
        c = self.costs.flatten()
        b = np.concatenate((self.demands, self.capacities))[:-1]
        s = c - self.apply_At(y)
        k = 0
        print(f"Initial value {self.value(x.reshape((self.N, self.M)))}")
        while True:
            mu = np.mean(np.dot(x, s))
            sigma = 1.0
            vec = (x * s - mu * sigma) / s
            # Compute y step
            d_y = vec
            d_y = self.apply_A(d_y)
            d_y = -self.apply_AGAt_inv(x / s, d_y)
            # Compute s step
            d_s = -self.apply_At(d_y)
            d_x = vec - x * d_s / s
            gamma_x = max(d_x / x)
            alpha_x = beta / gamma_x
            gamma_s = max(d_s / s)
            alpha_s = beta / gamma_s
            if (s - alpha_s * d_s < 0).any():
                import pdb; pdb.set_trace()
            x = x - alpha_x * d_x
            s = s - alpha_s * d_s
            y = y - alpha_s * d_y
            residual = np.dot(x, s)
            print(f"Step {alpha_x}, {alpha_s}")
            print(f"Min primal {x.min()}, min dual {s.min()}")
            value = self.value(x.reshape((self.N, self.M)))
            print(f"Iter {k+1}: value {value}, residual {residual}")
            k += 1

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
