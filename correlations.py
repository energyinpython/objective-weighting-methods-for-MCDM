import numpy as np


# spearman coefficient rs
def spearman(R, Q):
    N = len(R)
    denominator = N*(N**2-1)
    numerator = 6*sum((R-Q)**2)
    rS = 1-(numerator/denominator)
    return rS


# weighted spearman coefficient rw
def weighted_spearman(R, Q):
    N = len(R)
    denominator = N**4 + N**3 - N**2 - N
    numerator = 6 * sum((R - Q)**2 * ((N - R + 1) + (N - Q + 1)))
    rW = 1 - (numerator / denominator)
    return rW


# rank similarity coefficient WS
def coeff_WS(R, Q):
    sWS = 0
    N = len(R)
    for i in range(N):
        sWS += 2**(-int(R[i]))*(abs(R[i]-Q[i])/max(abs(R[i] - 1), abs(R[i] - N)))
    WS = 1 - sWS
    return WS


# pearson coefficient
def pearson_coeff(R, Q):
    numerator = np.sum((R - np.mean(R)) * (Q - np.mean(Q)))
    denominator = np.sqrt(np.sum((R - np.mean(R))**2) * np.sum((Q - np.mean(Q))**2))
    corr = numerator / denominator
    return corr


# kendall rank correlation coefficient
def kendall(R, Q):
    N = len(R)
    Ns, Nd = 0, 0
    for i in range(1, N):
        for j in range(i):
            if ((R[i] > R[j]) and (Q[i] > Q[j])) or ((R[i] < R[j]) and (Q[i] < Q[j])):
                Ns += 1
            elif ((R[i] > R[j]) and (Q[i] < Q[j])) or ((R[i] < R[j]) and (Q[i] > Q[j])):
                Nd += 1

    tau = (Ns - Nd) / ((N * (N - 1))/2)
    return tau

# goodman kruskal coefficient
def goodman_kruskal(R, Q):
    N = len(R)
    Ns, Nd = 0, 0
    for i in range(1, N):
        for j in range(i):
            if ((R[i] > R[j]) and (Q[i] > Q[j])) or ((R[i] < R[j]) and (Q[i] < Q[j])):
                Ns += 1
            elif ((R[i] > R[j]) and (Q[i] < Q[j])) or ((R[i] < R[j]) and (Q[i] > Q[j])):
                Nd += 1

    coeff = (Ns - Nd) / (Ns + Nd)
    return coeff