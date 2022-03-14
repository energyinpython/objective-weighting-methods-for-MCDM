import numpy as np

# reverse = True: descending order (TOPSIS, CODAS), False: ascending order (VIKOR, SPOTIS)
def rank_preferences(pref, reverse = True):
    """
    Rank alternatives according to MCDM preference function values.

    Parameters
    ----------
        pref : ndarray
            vector with MCDM preference function values for alternatives
        reverse : bool
            Boolean variable which is True for MCDM methods which rank alternatives in descending
            order and False for MCDM methods which rank alternatives in ascending
            order

    Returns
    -------
        ndarray
            vector with alternatives ranking. Alternative with 1 value is the best.
    """
    rank = np.zeros(len(pref))
    sorted_pref = sorted(pref, reverse = reverse)
    pos = 1
    for i in range(len(sorted_pref) - 1):
        ind = np.where(sorted_pref[i] == pref)[0]
        rank[ind] = pos
        if sorted_pref[i] != sorted_pref[i + 1]:
            pos += 1
    ind = np.where(sorted_pref[i + 1] == pref)[0]
    rank[ind] = pos
    return rank.astype(int)
