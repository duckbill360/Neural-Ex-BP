# Neural Ex-BP

import polar_codes
import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############### Polar Codes PARAMETERS ###############
iter_num = 2
N = 1024        # code length
n = int(np.log2(N))
layers_per_iter = 2 * n - 1
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
    hidden_layers = [tf.Variable(dtype=tf.float64, initial_value=tf.zeros((N, ), dtype=tf.float64),
                                 name='hidden_layer_' + str(i), trainable=False)
                     for i in range((2 * n - 1) * iter_num)]
    L = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num * n, N,), dtype=tf.float64),
                    name='L', trainable=False)
    R = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num * (n - 1), N,), dtype=tf.float64),
                    name='R', trainable=False)
    R_LLR = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((N, ), dtype=tf.float64),
                        name='R_LLR', trainable=False)
    y_hat = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((N, ), dtype=tf.float64),
                        name='y_hat', trainable=False)

    # k iterations
    # for k in range(iter_num):
    #     # R propagation first, L propagation later
    #     print('iter_num', k)
    #     # R propagation
    #     print('R')
    #     for i in range(k * layers_per_iter, k * layers_per_iter + (n - 1)):
    #         print(i)
    #         for j in range(N // 2):
    #             pass
    #             # TODO
    #
    #     # L propagation
    #     print('L')
    #     for i in range(k * layers_per_iter + (n - 1), (k + 1) * layers_per_iter):
    #         print(i)
    #         for j in range(N // 2):
    #             pass
    #             # TODO

    # Run through all hidden layers.
    # for i in range(len(hidden_layers)):
    #     indicator = i % (2 * n - 1)
    #     if indicator < (n - 1):     # -----------> former half: R propagation
    #         pass
    #     elif indicator >= (n - 1):  # -----------> later half: L propagation
    #         pass

    for i in hidden_layers:
        print(i)

    # Return a length-N vector
    return y_hat


# Min-Sum Algorithm
def f(x, y):
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

    # y_hat is a length-N vector.
    y_hat = forwardprop(x, alpha, beta)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        alpha_val = sess.run(alpha)
        beta_val = sess.run(beta)

    sess.close()
