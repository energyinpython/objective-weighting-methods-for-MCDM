import numpy as np

from normalizations import *
from mcdm_method import *


class TOPSIS(MCDM_method):
    def __init__(self, normalization_method = minmax_normalization):
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types):
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