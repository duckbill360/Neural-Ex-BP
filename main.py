# Neural Ex-BP

import polar_codes
import numpy as np
import tensorflow as tf

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############### Polar Codes PARAMETERS ###############
iter_num = 2
N = 1024        # code length
n = int(np.log2(N))
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 2.0
######################################################

Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10.0))
sigma = pow(Var, 1 / 2)

B_N = polar_codes.permutation_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
G = polar_codes.generate_G_N(N)


def forwardprop(x, alpha, beta):

    # L messages and R messages
    L = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num * n, N,), dtype=tf.float64),
                    expected_shape=(iter_num * n, N,), name='L')
    R = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num * n, N,), dtype=tf.float64),
                    expected_shape=(iter_num * n, N,), name='R')

    for k in range(iter_num):
        # L Propagation
        for i in range(iter_num * n + 10):
            for j in range(N // 2):
                pass
                # TODO

    print(L)
    print(R)


def f():
    return tf.sign(x) * tf.sign(y) * tf.minimum(tf.abs(x), tf.abs(y))

if __name__ == '__main__':

    # x is the input vector; y is the output vector
    # Their length is N.
    x = tf.placeholder(tf.float64, shape=(N, ), name='x')
    y = tf.placeholder(tf.float64, shape=(N, ), name='y')

    # alpha and beta are the weights of the Ex-BP decoder.
    # Can also be initialized using "random_normal".
    alpha = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num, N // 2, ), dtype=tf.float64),
                        expected_shape=(iter_num, N // 2, ), name='alpha', trainable=True)
    beta = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num, N // 2, ), dtype=tf.float64),
                       expected_shape=(iter_num, N // 2, ), name='beta', trainable=True)

    y_hat = forwardprop(x, alpha, beta)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        alpha_val = sess.run(alpha)
        beta_val = sess.run(beta)

        print(alpha)
        print(beta)

    sess.close()
