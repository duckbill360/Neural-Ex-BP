# Neural Ex-BP

import polar_codes
import numpy as np

############### PARAMETERS ###############
Times = 10      # i7 -> 50 : 10'25
N = 1024        # code length
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 2.0

Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10.0))
sigma = pow(Var, 1 / 2)

B_N = polar_codes.permutation_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
G = polar_codes.generate_G_N(N)

