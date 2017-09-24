# Neural Ex-BP

import polar_codes
import numpy as np
import tensorflow as tf

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############### Polar Codes PARAMETERS ###############
iter_num = 2
N = 64        # code length
n = int(np.log2(N))
layers_per_iter = 2 * n - 2     # Need an additional L layer.
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 2.0
######################################################

Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10.0))
sigma = pow(Var, 1 / 2)

B_N = polar_codes.permutation_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
G = polar_codes.generate_G_N(N)


def forwardprop(x_LLR, alpha, beta):
    # L messages and R messages
    hidden_layers = [tf.Variable(dtype=tf.float64, initial_value=tf.zeros((N, ), dtype=tf.float64),
                                 name='hidden_layer_' + str(i), trainable=False)
                     for i in range(layers_per_iter * iter_num + 1)]
    R_LLR = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((N, ), dtype=tf.float64),
                        name='R_LLR', trainable=False)
    y_hat = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((N, ), dtype=tf.float64),
                        name='y_hat', trainable=False)
    zero_float = tf.constant(0, dtype=tf.float64)

    # Initialize R_LLR.
    for index in frozen_indexes:
        R_LLR[index].assign(1000000)

    #####################################################
    ############## The first R Propagation ##############
    # The first R hidden layer, so take R_LLR as an input, and view L as 0
    print(0)
    for j in range(N // 2):
        hidden_layers[0][2 * j].assign(f(R_LLR[j], R_LLR[j + N // 2] + zero_float))
        hidden_layers[0][2 * j + 1].assign(f(zero_float, R_LLR[j]) + R_LLR[j + N // 2])
    # Not the first R hidden layer, so take the previous layers as inputs and view L as 0
    for i in range(1, n - 1):
        print(i)
        for j in range(N // 2):
            hidden_layers[i][2 * j].assign(f(hidden_layers[i - 1][j], hidden_layers[i - 1][j + N // 2] + zero_float))

            hidden_layers[i][2 * j + 1].assign(f(zero_float, hidden_layers[i - 1][j]) + hidden_layers[i - 1][j + N // 2])
    #####################################################
    #####################################################


    #####################################################
    #####################################################
    # Run through all the other hidden layers.
    for i in range(n - 1, len(hidden_layers) - 1):
        RL_indicator = i % layers_per_iter

        ############### R propagation ###############
        # Not the first 'R' propagation, need to consider the actual L messages.
        if RL_indicator < (n - 1):
            print('R')
            index = RL_indicator % (n - 1)
            print(index)
            if index == 0:
                for j in range(N // 2):
                    hidden_layers[i][2 * j].assign(f(R_LLR[j], R_LLR[j + N // 2]
                                                     + hidden_layers[i - 1 - 2 * index][2 * j + 1]))
                    hidden_layers[i][2 * j + 1].assign(f(hidden_layers[i - 1 - 2 * index][2 * j], R_LLR[j])
                                                       + R_LLR[j + N // 2])
            else:
                for j in range(N // 2):
                    hidden_layers[i][2 * j].assign(f(hidden_layers[i - 1][j], hidden_layers[i - 1][j + N // 2]
                                                     + hidden_layers[i - 1 - 2 * index][2 * j + 1]))
                    hidden_layers[i][2 * j + 1].assign(f(hidden_layers[i - 1 - 2 * index][2 * j],
                                                         hidden_layers[i - 1][j]) + hidden_layers[i - 1][j + N // 2])

        ############### L propagation ###############
        # Need to consider the previous R messages.
        elif RL_indicator >= (n - 1):
            print('L')
            index = RL_indicator % (n - 1)
            print(index)
            # If it is the first L layer
            if index == 0:
                for j in range(N // 2):
                    hidden_layers[i][j].assign(f(x_LLR[2 * j], x_LLR[2 * j + 1] + hidden_layers[i - 1][j + N // 2]))
                    hidden_layers[i][j + N // 2].assign(f(hidden_layers[i - 1][j], x_LLR[2 * j]) + x_LLR[2 * j + 1])
            else:
                for j in range(N // 2):
                    hidden_layers[i][j].assign(f(hidden_layers[i - 1][2 * j], hidden_layers[i - 1][2 * j + 1] +
                                                 hidden_layers[i - 1 - 2 * index][j + N // 2]))
                    hidden_layers[i][j + N // 2].assign(f(hidden_layers[i - 1 - 2 * index][j], hidden_layers[i - 1][2 * j])
                                                        + hidden_layers[i - 1][2 * j + 1])

    # Need an additional L layer (the last layer) to produce outputs
    last_layer_index = layers_per_iter * iter_num
    for j in range(N // 2):
        hidden_layers[last_layer_index][j].assign(f(hidden_layers[last_layer_index - 1][2 * j],
                                                    hidden_layers[last_layer_index - 1][2 * j + 1] + R_LLR[j + N // 2]))
        hidden_layers[last_layer_index][j + N // 2].assign(f(R_LLR[j], hidden_layers[last_layer_index - 1][2 * j])
                                                           + hidden_layers[last_layer_index - 1][2 * j + 1])
    #####################################################
    #####################################################


    # Return a length-N vector
    return y_hat


# Min-Sum Algorithm
def f(x, y):
    return tf.sign(x) * tf.sign(y) * tf.minimum(tf.abs(x), tf.abs(y))


if __name__ == '__main__':

    # x is the input vector; y is the output vector (Length: N)
    x = tf.placeholder(tf.float64, shape=(N, ), name='x')
    y = tf.placeholder(tf.float64, shape=(N, ), name='y')

    # alpha and beta are the weights to be trained of the Ex-BP decoder.
    # Can also be initialized using "random_normal".
    alpha = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num, N, ), dtype=tf.float64),
                        expected_shape=(iter_num, N // 2, ), name='alpha', trainable=True)
    beta = tf.Variable(dtype=tf.float64, initial_value=tf.zeros((iter_num, N, ), dtype=tf.float64),
                       expected_shape=(iter_num, N // 2, ), name='beta', trainable=True)

    # y_hat is a length-N vector.
    y_hat = forwardprop(x, alpha, beta)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        alpha_val = sess.run(alpha)
        beta_val = sess.run(beta)

    # Play a sound when the process finishes.
    import winsound
    duration = 1000  # millisecond
    freq = 440  # Hz
    winsound.Beep(freq, duration)
