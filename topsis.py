import numpy as np

from normalizations import *
from mcdm_method import *


class TOPSIS(MCDM_method):
    def __init__(self, normalization_method = minmax_normalization):
        """
        Create TOPSIS method object and select normalization method `normalization_method`.
        """
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 
        """
        TOPSIS._verify_input_data(matrix, weights, types)
        return TOPSIS._topsis(matrix, weights, types, self.normalization_method)

    @staticmethod
    def _topsis(matrix, weights, types, normalization_method):
        # Normalize matrix using chosen normalization (for example linear normalization)
        norm_matrix = normalization_method(matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights

        # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
        pis = np.max(weighted_matrix, axis=0)
        nis = np.min(weighted_matrix, axis=0)

        # Calculate chosen distance of every alternative from PIS and NIS
        Dp = (np.sum((weighted_matrix - pis)**2, axis = 1))**0.5
        Dm = (np.sum((weighted_matrix - nis)**2, axis = 1))**0.5

        return Dm / (Dm + Dp)