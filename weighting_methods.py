import numpy as np
from correlations import *
from normalizations import *
import copy
import sys
import itertools

# for i, j in itertools.product(range(x), range(y)):
# for (i,j) in [(i,j) for i in range(x) for j in range(y)]


# equal weighting
def equal_weighting(X, types):
    N = np.shape(X)[1]
    return np.ones(N) / N


# entropy weighting
def entropy_weighting(X, types):
    """
    Calculate criteria weights using objective Entropy weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray

    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    pij = np.abs(pij)
    m, n = np.shape(pij)

    H = np.zeros((m, n))

    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))
    d = 1 - h
    w = d / (np.sum(d))
    return w


# standard deviation weighting
def std_weighting(X, types):
    """
    Calculate criteria weights using objective Standard deviation weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # stdv = np.std(X, axis = 0)
    stdv = np.sqrt((np.sum(np.square(X - np.mean(X, axis = 0)), axis = 0)) / (X.shape[0]))
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(X, types):
    """
    Calculate criteria weights using objective CRITIC weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    correlations = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        correlations[i, j] = pearson_coeff(x_norm[:, i], x_norm[:, j])

    difference = 1 - correlations
    C = std * np.sum(difference, axis = 0)
    w = C / np.sum(C)
    return w


# gini weighting
def gini_weighting(X, types):
    """
    Calculate criteria weights using objective Gini coefficient-based weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    m, n = np.shape(X)
    G = np.zeros(n)
    # iteration over criteria j = 1, 2, ..., n
    for j in range(0, n):
        # iteration over alternatives i = 1, 2, ..., m
        Yi = np.zeros(m)
        if np.mean(X[:, j]) != 0:
            for i, k in itertools.product(range(m), range(m)):
                Yi[i] += np.abs(X[i, j] - X[k, j]) / (2 * m**2 * (np.sum(X[:, j]) / m))
        else:
            for i, k in itertools.product(range(m), range(m)):
                Yi[i] += np.abs(X[i, j] - X[k, j]) / (m**2 - m)

        G[j] = np.sum(Yi)
    return G / np.sum(G)


# MEREC weighting
def merec_weighting(matrix, types):
    """
    Calculate criteria weights using objective MEREC weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    X = copy.deepcopy(matrix)
    m, n = X.shape
    X = np.abs(X)
    norm_matrix = np.zeros(X.shape)
    norm_matrix[:, types == 1] = np.min(X[:, types == 1], axis = 0) / X[:, types == 1]
    norm_matrix[:, types == -1] = X[:, types == -1] / np.max(X[:, types == -1], axis = 0)
    
    S = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_matrix)), axis = 1)))
    Sp = np.zeros(X.shape)

    for j in range(n):
        norm_mat = np.delete(norm_matrix, j, axis = 1)
        Sp[:, j] = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_mat)), axis = 1)))

    E = np.sum(np.abs(Sp - S.reshape(-1, 1)), axis = 0)
    w = E / np.sum(E)
    return w


# statistical variance weighting
def stat_var_weighting(X, types):
    """
    Calculate criteria weights using objective Statistical variance weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    criteria_type = np.ones(np.shape(X)[1])
    xn = minmax_normalization(X, criteria_type)
    v = np.mean(np.square(xn - np.mean(xn, axis = 0)), axis = 0)
    # vv = np.var(xn, axis = 0)
    w = v / np.sum(v)
    return w


# CILOS weighting
def cilos_weighting(X, types):
    """
    Calculate criteria weights using objective CILOS weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    xr = copy.deepcopy(X)
    # convert negative criteria to positive criteria
    xr[:, types == -1] = np.min(X[:, types == -1], axis = 0) / X[:, types == -1]
    # normalize DM
    xn = xr / np.sum(xr, axis = 0)
    
    # calculate square matrix
    A = xn[np.argmax(xn, axis = 0), :]
    
    # calculate relative impact loss matrix
    pij = np.zeros((X.shape[1], X.shape[1]))
    for j, i in itertools.product(range(X.shape[1]), range(X.shape[1])):
        pij[i, j] = (A[j, j] - A[i, j]) / A[j, j]

    F = np.diag(-np.sum(pij - np.diag(np.diag(pij)), axis = 0)) + pij
    AA = np.zeros(F.shape[0])
    AA[0] = sys.float_info.epsilon
    q = np.linalg.inv(F).dot(AA)
    return q / np.sum(q)


# IDOCRIW weighting
def idocriw_weighting(X, types):
    """
    Calculate criteria weights using objective IDOCRIW weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    q = entropy_weighting(X, types)
    w = cilos_weighting(X, types)
    weights = (q * w) / np.sum(q * w)
    return weights


# angle weighting
def angle_weighting(X, types):
    """
    Calculate criteria weights using objective Angle weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    m, n = X.shape
    X = sum_normalization(X, types)
    B = np.ones(m) * (1 / m)
    u = np.arccos(np.sum(X / m, axis = 0) / (np.sqrt(np.sum(X ** 2, axis = 0)) * np.sqrt(np.sum(B ** 2))))
    w = u / np.sum(u)
    return w


# coeffcient of variance weighting
def coeff_var_weighting(X, types):
    """
    Calculate criteria weights using objective Coefficient of variance weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    m, n = X.shape
    criteria_types = np.ones(n)
    B = sum_normalization(X, criteria_types)
    Bm = np.sum(B, axis = 0) / m

    std = np.sqrt(np.sum(((B - Bm)**2), axis = 0) / (m - 1))
    ej = std / Bm
    w = ej / np.sum(ej)
    return w