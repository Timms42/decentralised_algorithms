import numpy as np
import scipy.sparse as sparse
import time
import networkx as nx
import numba

def alg_pdtr(z0, JA, B, W, tau: float, residue, relerror, max_iters=1000, data=None):
    """
    :param z0: (N, N * (xd + yd)) array, initial iterate
    :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
    :param B: function handle for computing forward operator. Called as B(z, data).
    :param tau: step size.
    :param data: optional dictionary of problem data, if necessary to compute JA() and B().
    :return:
    """
    # Initialise error
    res = []
    rel = []

    N = W.shape[0]
    tW = 0.5 * (np.eye(N) + W)
    start = time.perf_counter()
    # Initialise variables
    v0 = B(z0, data)
    u1 = z0 - tau * v0
    z1 = JA(u1, tau, data)

    # Main loop
    for k in range(max_iters):
        v1 = 2 * B(z1, data) - B(z0, data)
        u2 = W @ z1 + u1 - tW @ z0 - tau * (v1 - v0)
        z2 = JA(u2, tau, data)

        res.append(residue(z2, z1, tau))
        rel.append(relerror(z2))

        z0 = z1.copy()
        z1 = z2.copy()
        v0 = v1.copy()
        u1 = u2.copy()

        print(f'PDTR: iteration {k + 1:04} of {max_iters}', end='\r')

    dur = time.perf_counter() - start
    return z2, rel, res, dur

def alg_midas(z1, JA, B, W, tau: np.array, residue, relerror, max_iters=1000, data=None, beta=None):
    r"""
    :param z1: initial iterate
    :param tau: (NxN array) step size matrix
    :param data: optional dictionary of problem data, if necessary to compute JA() and B().
    :param beta: Must satisfy I-(\beta/2)*sqrt(\tau)@(I-W)@sqrt(\tau)>0. Default \beta = 1/max(\tau).
    :return:
    """

    # Initialise error
    res = []
    rel = []

    N = W.shape[0]
    if beta is None:
        beta = 0.9 / np.max(tau)
    tW = np.eye(N) - beta/2 * tau @ (np.eye(N) - W)

    start = time.perf_counter()

    # Initialisation
    y1 = np.zeros_like(z1)      # y1 = y^0
    v1 = B(y1, data)            # v1=B(y^0)
    x1 = JA(z1, tau, data)      # x^1=J_{\tau A}(z^1)
    y2 = 2 * x1 - z1 - tau @ v1
    z2 = z1 + y2 - x1

    z1 = z2.copy()
    v0 = v1.copy()
    x0 = x1.copy()
    y0 = y1.copy()
    y1 = y2.copy()

    # Main loop
    for k in range(max_iters):
        v1 = 2 * B(y1, data) - B(y0, data)
        x1 = JA(z1, tau, data)
        z2 = z1 - x1 + tW @ (2*x1 - x0 - tau @ (v1 - v0))
        y2 = x1 + z2 - z1

        res.append(residue(x1, x0, tau))
        rel.append(relerror(x1))

        z1 = z2.copy()
        v0 = v1.copy()
        y0 = y1.copy()
        y1 = y2.copy()
        x0 = x1.copy()
          
        print(f'MIDAS: iteration {k + 1:04} of {max_iters}', end='\r')

    
    dur = time.perf_counter() - start
    return x1, rel, res, dur

# Auxiliary functions for boosted algorithm
def boosted_initial(G, m):
    """Create initial iterate for boosted algorithm"""
    N = len(list(G))
    Adj = nx.adjacency_matrix(G).toarray()  # Adjacency matrix

    # Set  up variable structure for z_ij array -- Construct matrix indicating valid indices for x_ij and y_ij variables
    # (Adj + I)^2 is nonzero if j=i, or j is a 1st or 2nd neighbour of i
    xvar_matrix = np.linalg.matrix_power(Adj.copy() + np.eye(N), 2)
    # Convert to 0/1
    xvar_matrix = sparse.csr_array((xvar_matrix > 0.5).astype(int))

    xvar_matrix = sparse.kron(xvar_matrix, np.ones((1, 4 * m)))  # Each agent tracks a 4m-variable for each 1st/2nd agent
    # Extract indices of non-zero elements in xvar_matrix, i.e. 1st and 2nd nbhrhood for each agent
    xvar_index = np.nonzero(xvar_matrix)
    xvar_values = np.zeros(int(np.sum(xvar_matrix)))

    # Initialise variables as sparse matrices, based on graph node neighbours.
    z1 = sparse.csr_array((xvar_values, (xvar_index[0], xvar_index[1])))

    return z1

def get_neighbours(G):
    """
    Compute set of neighbours and 2nd neighbours for each agent
    :param G: agent network graph, networkx.Graph
    :return: numba.typed.List for compatibility with @numba.njit
    """

    N = len(list(G))
    neighbours = [numba.typed.List(sorted(list(G.neighbors(i)) + [i])) for i in range(N)]     # INCLUDES AGENT I

    # Note: intersection of neighbours[i] and two_neighbours[i] is empty
    two_neighbours = [
        numba.typed.List(sorted(list(set(n2 for j in neighbours[i] for n2 in list(G.neighbors(j)) if j != i if n2 != i))))
        for i in range(N)]

    return numba.typed.List(neighbours), numba.typed.List(two_neighbours)

@numba.njit
def sigma(x, y, mix, nbhr):
    r"""
    Compute \sigma_{ik} variables
    :param x: current value of x0 in boosted MIDAS
    :param y: current value of mu
    :param nbhr: set of neighbours for each agent
    :param mix: mixing matrix
    :return:
    """

    N, m = y.shape[0], int(y.shape[1]/2)
    sigma0 = np.zeros_like(y)

    for ii in range(N):  # Compute entries of sigma0
        sigma_ll_sum = np.zeros(2 * m)

        for ll in range(N):
            x_lnotini = np.zeros(2 * m)
            x_inotinl = np.zeros(2 * m)

            lnotini = set(nbhr[ll]) - set(nbhr[ii])
            inotinl = set(nbhr[ii]) - set(nbhr[ll])

            if len(lnotini) != 0:
                # If this set is nonempty, else leave vector as 0s
                for jj in lnotini:
                    # Create sum over agents
                    x_lnotini += x[ll, 2 * m * jj:2 * m * (jj + 1)]

            if len(inotinl) != 0:
                # If this set is nonempty, else leave vector as 0s
                for jj in inotinl:
                    x_inotinl += x[ll, 2 * m * jj:2 * m * (jj + 1)]

            # Create sum over ll in nbhr[ii]
            sigma_ll_sum += mix[ii, ll] * (y[ll] + x_lnotini - x_inotinl)

        sigma0[ii, :] = sigma_ll_sum.copy()
    return sigma0

def mu_init(y, nbhr):
    """
    :param y: scipy.sparse.csr_array
    :param nhbr: numba.Typed.List
    """
    # Construct mu^0_i = \sum_{j\notin nhbr[i]} y^0_{ij}
    N, m = y.shape[0], int(y.shape[1]/y.shape[0]/4)
    mu0 = sparse.lil_array(np.zeros((N, 2*m)))
    for ii in range(N):
        yi = y[[ii],:].toarray().flatten()
        mu0[ii,:] = np.sum([yi[2*m*jj:2*m*(jj+1)] for jj in set(range(N))-set(nbhr[ii])], axis=0).flatten()
    return mu0


def alg_boost(z1, JA, B, W, G, tau: np.array, residue, relerror, max_iters=1000, data=None, beta=None):
    """
    :param z1: (N, N * 2 * m) array, initial iterate
    :param JA: function handle for resolvent, called as JA(z_k, tau, data)
    :param B: function handle for forward operator, called as B(y_k, mu_k, nbhrs, data)
    :param tau: (NxN sparse array) step size matrix
    :param data: optional dictionary of problem data, if necessary to compute JA() and B().
    :return: epsilon-optimality error, residuals, total time
    """
    # Initialise error
    rel = np.zeros(max_iters)
    res = np.zeros(max_iters)

    N = W.shape[0]
    m = int(z1.shape[1]/N/4)

    if beta is None:
        beta = 0.9 / np.max(tau)
    tW = sparse.eye_array(N)-  beta/2 * tau.dot(sparse.eye_array(N) - W)
    nbhr, _ = get_neighbours(G)

    start = time.perf_counter()

    # Initialise variables
    y1 = sparse.csr_array((np.zeros(z1.shape)))
    mu1 = mu_init(y1, nbhr)  # mu^{0}
    v1 = B(y1, mu1, nbhr, data)
    x1 = JA(z1, tau, data)
    y2 = 2 * x1 - z1 - tau @ v1
    z2 = z1 + y2 - x1
    mu2 = mu_init(y2, nbhr)
    sigma1 = sparse.csr_array(sigma(y2.toarray(), mu2.toarray(), tW.toarray(), nbhr))    # sigma1 = \sum (\mu^0 + \sum y^0 + ...)

    z1 = z2.copy()  # z1=z^1
    v0 = v1.copy()  # v0=B(z^0)=2*B(z^0)-B(z^{-1})
    x0 = x1.copy()
    y0 = y1.copy()
    y1 = y2.copy()
    mu0 = mu1.copy()  # mu_neg1 = \mu^{-1}
    mu1 = mu2.copy()
    sigma0 = sigma1.copy()  # sigma_neg1 = \sigma^{-1}

    for k in range(max_iters):
        v1 = 2 * B(y1, mu1, nbhr, data) - B(y0, mu0, nbhr, data)
        x1 = JA(z1, tau, data)
        z2 = z1 - x1 + tW.dot(2*x1 - x0 - tau @ (v1 - v0))
        y2 = x1 + z2 - z1
        sigma1 = sparse.csr_array(sigma(y1.toarray(), mu1.toarray(), tW.toarray(), nbhr))
        mu2 = 2*sigma1 - sigma0

        res[k] = residue(x1, x0, tau)
        rel[k] = relerror(x1)

        mu0 = mu1.copy()
        mu1 = mu2.copy()
        sigma0 = sigma1.copy()
        z1 = z2.copy()
        v0 = v1.copy()
        y0 = y1.copy()
        y1 = y2.copy()
        x0 = x1.copy()

        print(f'Boost: iteration {k + 1:04} of {max_iters}', end='\r')
    dur = time.perf_counter() - start

    return x1, rel, res, dur