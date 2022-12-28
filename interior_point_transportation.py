import numpy as np


class TransportationProblem:
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
        Xs = X.sum(axis=1)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = X.sum(axis=0)
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
        A = TransportationProblem.make_dense_A(N, M)
        return A @ np.diag(G) @ A.transpose()

    def apply_A(N: int, M: int, t):
        """
        Apply A to the vector t without forming it explicitly
        """
        assert t.ndim == 1
        assert t.size == N * M
        X = t.reshape((N, M))
        Xs = X.sum(axis=1)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = X.sum(axis=0)
        return np.concatenate((Xs, Xt))

    def apply_AGAt(N: int, M: int, G, t):
        """
        Apply AGAt to the vector t without forming it explicitly
        """
        assert G.ndim == 1
        assert G.size == N * M
        assert t.size == N + M - 1
        X = G.reshape((N, M))
        Xs = X.sum(axis=1)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = X.sum(axis=0)
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
        Xs = X.sum(axis=1)
        X = X[:, :-1]  # Redundant constraint removed
        Xt = X.sum(axis=0)

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
