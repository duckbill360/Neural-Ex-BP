# Neural Ex-BP

import polar_codes
import numpy as np
import tensorflow as tf

############### Polar Codes PARAMETERS ###############
iter_num = 2
N = 1024        # code length
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 2.0
######################################################

Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10.0))
sigma = pow(Var, 1 / 2)

B_N = polar_codes.permutation_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
G = polar_codes.generate_G_N(N)

if __name__ == '__main__':

    x = tf.placeholder(tf.float64, shape=(N, ), name='x')
    y = tf.placeholder(tf.float64, shape=(N, ), name='y')

    alpha = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num, N // 2, ), dtype=tf.float64), expected_shape=(iter_num, N // 2, ), name='alpha', trainable=True)
    beta = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num, N // 2, ), dtype=tf.float64), expected_shape=(iter_num, N // 2, ), name='beta', trainable=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    alpha_val = sess.run(alpha)
    beta_val = sess.run(beta)
    print(alpha_val)
    print(beta_val)

    print(alpha)
    print(beta)

    sess.close()
