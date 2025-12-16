"""
Set up the VPP problem from Tam, Timms, and Zhang, and solve using Algorithm 1, Algorithm 2, and PDTR.
Algorithm 2 performance improved with Numba.
"""
import numpy as np
import cvxopt
import networkx as nx
import pickle
import scipy.sparse as sparse
import algorithms
import matplotlib
import matplotlib.pyplot as plt
# from plots import plot_results

def prox(tau, agent: int, y: np.ndarray, data):
    r"""
    Define function for computing prox_{\tau*g_i} of ux_k[ii] (the primal component)
    :param tau: step size
    :param agent: agent number 0,...,N-1. Determines the function g_i to compute proximal operator for.
    """

    m = int(len(y)/2)
    if isinstance(tau, float):
        step = tau
    else:   # tau is a matrix
        step = tau[agent, agent]

    l_up_vec = data['l_up'][agent]
    l_low_vec = data['l_low'][agent]
    Qplus_mat = data['Q+'][agent]
    Qminus_mat = data['Q-'][agent]
    pplus_vec = data['p+'][agent]
    pminus_vec = data['p-'][agent]
    eplus = data['e+'][agent]
    eminus = data['e-'][agent]

    Q_mat = np.block([[Qplus_mat, np.zeros(Qminus_mat.shape)],
                      [np.zeros(Qplus_mat.shape), Qminus_mat]])
    p_vec = np.block([pplus_vec, pminus_vec])

    # Gx <= h
    G = np.block([
        [np.block([[np.identity(m), np.zeros((m, m))], [np.zeros((m, m)), eminus * np.identity(m)]])],  # x<=xbar
        [-np.identity(2 * m)],  # -x <= 0
        [np.block([eplus * np.tril(np.ones((m, m))), -(1 / eminus) * np.tril(np.ones((m, m)))])],
        # charge state <= l_up
        [np.block([-1 * eplus * np.tril(np.ones((m, m))), (1 / eminus) * np.tril(np.ones((m, m)))])]
        # -charge state <= -l_low
    ])

    h = np.block([[np.array(data['x_bar+'][agent])],
                  [np.array(data['x_bar-'][agent])],
                  [np.zeros((2 * m, 1))],
                  [l_up_vec],
                  [-1 * l_low_vec]])

    ## transform the np.arrays into cvxopt.matrix
    P = cvxopt.matrix(Q_mat + (1. / step) * np.identity(np.shape(Q_mat)[0]))
    q = cvxopt.matrix(p_vec - (1. / step) * y)
    h = cvxopt.matrix(h)
    G = cvxopt.matrix(G)

    output_dict = cvxopt.solvers.qp(P, q, G, h, options={"show_progress": False})
    output = output_dict["x"]
    if output_dict['status'] == 'unknown':
        raise ValueError(f"Infeasible for agent {agent}")

    return np.array(output)


def JA(z, par_tau, data):
    """
    :param z: variable
    :param par_tau: step size
    :param m: time periods
    :param N: agents
    :param data: dictionary of model parameters
    :return:
    """

    JAz = z.copy()  # np.zeros_like(z)
    N = z.shape[0]
    m = int(z.shape[1]/N/4)

    for ii in range(N):
        zi = np.array(z[ii]).flatten()
        # Extract x and y variables
        xi, yi = zi[:2 * m * N].reshape(N, 2 * m), zi[2 * m * N:].reshape(N, 2 * m)

        # Compute resolvent of x and y components
        JAxii = prox(par_tau, ii, xi[ii], data)
        JAyii = np.clip(yi[ii], 0, None)

        # Insert resolvent components into array
        JAz[ii, 2 * m * ii:2 * m * (ii + 1)] = JAxii.ravel()
        JAz[ii, 2 * m * (ii + N):2 * m * (ii + 1 + N)] = JAyii

    return JAz


def B(z, data):
    r""" Evaluate operator \mathbf{B}(z) = B_1(z[1]) x ... x B_N(z[N])"""
    N = z.shape[0]
    m = int(z.shape[1] / N / 4)

    b_vec = data['b']  # K-d
    c_vec = data['c']  # objective coefficients
    d_vec = data['d']
    emin_vec = data['e-']
    Amat = np.block([[np.identity(m)],
                     [-np.identity(m)]])

    Bz = np.zeros_like(z)
    for ii in range(N):
        # Extract and reshape x^+, x^- and y^+, y^- components
        zi = np.array(z[ii]).flatten()
        xi, yi = zi[:2 * m * N].reshape(N, 2 * m), zi[2 * m * N:].reshape(N, 2 * m)

        # Compute sum(x_j^+ - e_j^- x_j^-)
        lin_combo = (1 / N) * np.sum(np.array([xi[jj, :m] - emin_vec[jj] * xi[jj, m:] for jj in range(N)]), axis=0)
        agg = np.sum([xi[jj] for jj in range(N)], axis=0)
        # objective = c_vec[ii] # c^T u_i
        objective = np.block([agg[:m] - agg[m:] + d_vec + xi[ii,:m] - xi[ii,m:], -agg[:m] + agg[m:] - d_vec - xi[ii,:m] + xi[ii,m:]]) # aggregate objective

        # Compute Bz for x and y components, agent i's own variables
        # Bxii = c_vec[ii] + (1 / N) * np.block([Amat.T @ yi[ii], -emin_vec[ii] * Amat.T @ yi[ii]])     
        Bxii = objective + np.block([Amat.T @ yi[ii], -emin_vec[ii] * Amat.T @ yi[ii]])
        Byii = np.block([b_vec[ii] - d_vec, d_vec]) - Amat @ lin_combo

        # Insert agent i's components of Bz into array, leaving other entries as 0
        Bz[ii, 2 * m * ii:2 * m * (ii + 1)] = Bxii
        Bz[ii, 2 * m * (ii + N):2 * m * (ii + 1 + N)] = Byii
    return Bz


def JAB(z, par_tau, data):
    r"""
    Boosted/sparse implementation of resolvent J_{\tau A}(z)
    """

    N = z.shape[0]
    m = int(z.shape[1]/N/4)

    JAz = sparse.lil_array(z.copy())

    for ii in range(N):
        # Convert array slice to dense array for manipulation
        zi = z[[ii], :].toarray().flatten()
        # Extract x and y variables
        xi, yi = zi[:2 * m * N].reshape(N, 2 * m), zi[2 * m * N:].reshape(N, 2 * m)

        # Compute resolvent of x and y components
        JAxii = prox(par_tau, ii, xi[ii], data)
        JAyii = np.clip(yi[ii], 0, None)

        # Insert resolvent components into array
        JAz[ii, 2 * m * ii:2 * m * (ii + 1)] = JAxii.ravel()
        JAz[ii, 2 * m * (ii + N):2 * m * (ii + 1 + N)] = JAyii

    return JAz.tocsr()


def BB(z, mu, nb, data):
    r"""
    Boosted/sparse implementation of boosted B operator
     \mathbf{B}(z) = B_1(z[1], mu[N]) x ... x B_N(z[N], mu[N])
    """
    N = z.shape[0]
    m = int(z.shape[1] / N / 4)

    b_vec = data['b']  # K-d
    c_vec = data['c']  # objective coefficients
    d_vec = data['d']
    Amat = np.block([[np.identity(m)],
                     [-np.identity(m)]])

    # Use lil_array for efficient slicing, then convert to csr_array at the end
    Bz = sparse.lil_array(z.shape)
    # sparse.csr_array((xvar_values, (xvar_index[0], xvar_index[1])))
    for ii in range(N):
        # Convert array slice to dense array for manipulation
        zi = z[[ii], :].toarray().flatten()
        # Extract and reshape x^+, x^- and y^+, y^- components
        xi, yi = zi[:2 * m * N].reshape(N, 2 * m), zi[2 * m * N:].reshape(N, 2 * m)
        # Extract mu_i variable
        mui = mu[[ii], :].toarray().flatten()
        agg = mui + np.sum([xi[jj] for jj in nb[ii]], axis=0)
        # objective = c_vec[ii] # c^T xi
        objective = np.block([agg[:m] - agg[m:] + d_vec + xi[ii,:m] - xi[ii,m:], -agg[:m] + agg[m:] - d_vec - xi[ii,:m] + xi[ii,m:]]) # aggregate objective

        # Compute Bz for x and y components, agent i's own variables
        Bxii = objective + np.block([Amat.T @ yi[ii], -Amat.T @ yi[ii]])
        Byii = np.block([b_vec[ii] - d_vec, d_vec]) - Amat @ (agg[:m] - agg[m:])

        # Insert agent i's components of Bz into array, leaving other entries as 0
        Bz[[ii], 2 * m * ii:2 * m * (ii + 1)] = Bxii
        Bz[[ii], 2 * m * (ii + N):2 * m * (ii + 1 + N)] = Byii

    # Return a csr_array sparse matrix
    return Bz.tocsr()

# Performance metrics
def residue(x, y, tau):
    """ Distance between iterates """
    if isinstance(tau, float):
        return np.linalg.norm(x - y)/tau
    elif isinstance(tau, np.ndarray):
        tauinvsqrt = np.divide(1, np.sqrt(tau), where=(tau!=0))
        return  np.linalg.norm(tauinvsqrt @(x - y))

def epserrorB(z, x, mu, tau, nbhr, data):
    r""" Epsilon-optimality error for boosted algorithm
    0 \approx\in (A+B)(x) <--> tau^{-1} * (z - x) + B(x) \approx 0
    :param x: output of JA(z)
    :param tau: float or (N, N) array
    """
    Ax1 = sparse.diags(1/tau.diagonal()) @ (z - x)
    return sparse.linalg.norm(Ax1 + BB(x, mu, nbhr, data))

def relerror(z, zsol, zsolnorm, d_index):
    """ Distance from reference solution to agent 0's iterate """
    N, m = zsol.shape[0], zsol.shape[1]/2

    return np.linalg.norm(z[d_index].reshape((N, -1)) - zsol) / zsolnorm

def charge_plot(xx_plus, xx_minus, data, m, N, algorithm: str, fontsz: list):
    """
    Create and show charging decision plot from data
    :param xx_plus: charging decisions
    :param xx_minus: discharging decisions
    :param dem: non-VPP demand vector
    :param cap: parameter b = K+max(d)
    :param m: time periods
    :param N: agents
    :param algorithm: string, name of algorithm used
    :param fontsz: [title fontsize, ticks fontsize, axis label fontsize]
    :param colors:
    :return:
    """

    dem = data['d']
    cap = data['b']
    titlesz, ticksz, axissz = fontsz
    agent_colours = matplotlib.colormaps['tab20b'](np.linspace(0, 1, N))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    # Plot charging and discharging decisions
    base_charge = np.zeros(m)
    base_discharge = np.zeros(m)
    for i in range(N):
        ax.bar(np.arange(0, m), xx_plus[i], label="Agent " + str(i + 1),
               bottom=base_charge, color=agent_colours[i])
        ax.bar(np.arange(0, m), -xx_minus[i], bottom=base_discharge, color=agent_colours[i])
        base_charge += xx_plus[i]
        base_discharge -= xx_minus[i]

    ax.plot(dem, 'r', label='Demand (shifted)', linewidth=3)
    ax.hlines(y=cap, xmin=-1, xmax=m, colors='k', label='Grid capacity', linewidth=3)

    ax.axhline(y=0, xmin=0, xmax=m, color='k')  # Draw solid line at y=0 to emphasise charging/discharging
    ax.set_xlim([-0.2, m])
    # ax.set_ylim([-1, np.max(xx_plus)])
    ax.legend(bbox_to_anchor=(1, 1))
    ax.grid()
    ax.set_xlabel("Time horizion (hour)", fontsize=axissz)
    ax.set_ylabel("Power (kW)", fontsize=axissz)
    ax.set_title(f'Charging decisions from {algorithm} for VPPs', fontsize=titlesz)
    ax.tick_params(axis='both', which='both', labelsize=ticksz)
    fig.show()

    return fig, ax


# Model dimension parameters
N_agents = 10
numt = 24
maxiters=1000
graph = 'cycle'  # ['barbell',  'bipartite', 'complete', 'cycle', 'grid2D']

# Load existing model parameter values
with open(f'C:\\Users\\TIMMSLF\OneDrive - The University of Melbourne\\Documents\\PhD\Code\\Data\\vpp_data_N{N_agents}_t24', 'rb') as f:
    data = pickle.load(f)

with open(f'C:\\Users\\TIMMSLF\\OneDrive - The University of Melbourne\\Documents\\PhD\\Code\\Data\\graph_{graph}_N{N_agents}', 'rb') as f:
    G = pickle.load(f)

diag_index = np.concatenate([np.repeat(jj, 2 * numt) for jj in range(N_agents)]), np.arange(2 * numt * N_agents, dtype=int)

results_dict = {}

Lap = nx.laplacian_matrix(G).todense()  # Laplacian
alpha = 0.505 * np.linalg.norm(Lap, ord=2)  # Want (1/2) * lambda_max < alpha <= (2/3) * lambda_max
W = np.eye(N_agents) - Lap / alpha
WB = sparse.csr_array(W)    # Sparse version of W for boosted algorithm

# Estimating Lipchitz constant of B operator -> a linear operator
# L = max(np.max(data['c']), np.max(data['b'])) # linear objective
L = 2*np.sqrt(2) #aggregate objective
tau_midas = np.diag(0.9 / (8 * L) * np.ones(N_agents))
tau_boost = sparse.csr_array(tau_midas)
tau_pdtr = 0.9 * (1 + np.min(np.linalg.eigvalsh(W))) / (2 * L)

betanorm = 0.9/np.linalg.norm(np.sqrt(tau_midas)@((np.eye(N_agents)-W)/2)@np.sqrt(tau_midas), ord=2)    # \beta = 1/||tau^(1/2)*((I-W)/2)*tau^(1/2))||

z0 = np.random.random((N_agents, 4*N_agents*numt)) # 24 time periods, x2 for charge and discharge, x2 for primal and dual
z0 *= 10/np.linalg.norm(z0)
# z_init_boost = algorithms.boosted_initial(G, numt)

# ---------------------------------------------------
# # MIDAS solution
sol_midas, _, results_dict['res_midas'], results_dict['time_midas'] = algorithms.alg_midas(z0, JA, B, W, tau_midas, residue, lambda z: 0,
                                                                 max_iters=maxiters, data=data, beta=betanorm)

# MIDASboost solution z1, JA, B, W, G, tau, data, residue, relerror, epserror, max_iters
(sol_boost, results_dict['err_rel_boost'], 
 results_dict['res_boost'], results_dict['time_boost']) = algorithms.alg_boost(z_init_boost, JAB, BB, WB,
                                                                    G, tau_boost,
                                                                    residue, lambda z: relerror(z, tseng_sol, tseng_solnorm, diag_index),
                                                                    max_iters=maxiters, data=data, beta=betanorm)

# PDTR solution
(sol_pdtr, results_dict['err_rel_pdtr'],
 results_dict['res_pdtr'], results_dict['time_pdtr']) = algorithms.alg_pdtr(z_init, JA, B, W, tau_pdtr,
                                                                 residue, lambda z: relerror(z, tseng_sol, tseng_solnorm, diag_index),
                                                                 max_iters=maxiters, data=data)

# plot_results(results_dict)

# Charge plot
# zsol = np.zeros((N_agents, 2*numt))
# for ii in range(N_agents):
#     zsol[ii] = sol_midas[ii,2*ii*numt:2*(ii+1)*numt]

# fig_charge, ax_charge = charge_plot(zsol[:,:numt], zsol[:,numt:], data, numt, N_agents, 'midas', [20,14, 18])
# plt.savefig('vpp_aggregate_charge.pdf')