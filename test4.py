# Neural Ex-BP

import polar_codes
import numpy as np
import tensorflow as tf

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############### Polar Codes PARAMETERS ###############
iter_num = 2
N = 8        # code length
n = int(np.log2(N))
layers_per_iter = 2 * n - 2     # Need an additional L layer.
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 3.0
######################################################

Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10.0))
sigma = pow(Var, 1 / 2)

B_N = polar_codes.permutation_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
G = polar_codes.generate_G_N(N)

sess = tf.Session()


def forwardprop(x_LLR, alpha, beta):

    # The list of all hidden layers
    hidden_layers = []

    # Initialize R_LLR
    LLR = [0 for i in range(N)]
    for i in frozen_indexes:
        LLR[i] = 1000000

    R_LLR = tf.constant(LLR, dtype=tf.float64)
    N_2_zeros = tf.constant(0, dtype=tf.float64, shape=(N // 2,))
    Ex_counter = 0

    #####################################################
    ############## The first R Propagation ##############
    # The first R hidden layer, so take R_LLR as an input, and view L as 0
    cur_idx = 0
    print('R Layer :', cur_idx)
    # for j in range(N // 2):
    #     hidden_layers[0][2 * j].assign(f(R_LLR[j], R_LLR[j + N // 2] + zero_float))
    #     hidden_layers[0][2 * j + 1].assign(f(zero_float, R_LLR[j]) + R_LLR[j + N // 2])
    upper = f(R_LLR[: N // 2], R_LLR[N // 2:] + N_2_zeros)
    lower = f(N_2_zeros, R_LLR[: N // 2]) + R_LLR[N // 2:]
    previous_layer = interlace(upper, lower)
    hidden_layers.append(previous_layer)

    # Not the first R hidden layer, so take the previous layers as inputs and view L as 0
    for i in range(n - 2):
        cur_idx += 1
        print("R Layer :", cur_idx)
        # for j in range(N // 2):
        #     hidden_layers[i][2 * j].assign(f(hidden_layers[i - 1][j], hidden_layers[i - 1][j + N // 2] + zero_float))
        #
        #     hidden_layers[i][2 * j + 1].assign(f(zero_float, hidden_layers[i - 1][j]) + hidden_layers[i - 1][j + N // 2])
        upper = f(previous_layer[: N // 2], previous_layer[N // 2:] + N_2_zeros)
        lower = f(N_2_zeros, previous_layer[: N // 2]) + previous_layer[N // 2:]
        previous_layer = interlace(upper, lower)
        hidden_layers.append(previous_layer)
    #####################################################
    #####################################################


    #####################################################
    #####################################################
    # Run through 'iter_num' times.
    # Each iteration consists of 1 full L propagation and 1 full R propagation.
    for k in range(iter_num - 1):
        ###### L Propagation
        # The first L propagation layer.
        cur_idx += 1
        print("L Layer :", cur_idx)
        upper = f(x_LLR[: N: 2], x_LLR[1: N: 2] + previous_layer[N // 2:])
        lower = f(previous_layer[: N // 2], x_LLR[: N: 2]) + x_LLR[1: N: 2]
        previous_layer = concatenate(upper, lower)

        # This is for the Ex-BP decoder.
        upper_Ex = alpha[Ex_counter, : N // 2] * upper + beta[Ex_counter, N // 2:] * previous_layer[N // 2:]
        lower_Ex = alpha[Ex_counter, N // 2:] * lower + beta[Ex_counter, : N // 2] * previous_layer[: N // 2]
        Ex_counter += 1
        previous_layer = concatenate(upper_Ex, lower_Ex)

        hidden_layers.append(previous_layer)

        # Not the first L propagation layer.
        for i in range(n - 2):
            cur_idx += 1
            print("L Layer :", cur_idx)
            corresponding_R_idx = cur_idx - (2 * i + 3)
            corresponding_R_layer = hidden_layers[corresponding_R_idx]
            upper = f(previous_layer[: N: 2], previous_layer[1: N: 2] + corresponding_R_layer[N // 2:])
            lower = f(corresponding_R_layer[: N // 2], previous_layer[: N: 2]) + previous_layer[1: N: 2]
            previous_layer = concatenate(upper, lower)
            hidden_layers.append(previous_layer)


        ###### R Propagation
        # The first R propagation layer.
        cur_idx += 1
        print("R Layer :", cur_idx)
        upper = f(R_LLR[: N // 2], R_LLR[N // 2:] + previous_layer[1: N: 2])
        lower = f(previous_layer[: N: 2], R_LLR[: N // 2]) + R_LLR[N // 2:]
        previous_layer = interlace(upper, lower)
        hidden_layers.append(previous_layer)

        # Not the first R propagation layer.
        for i in range(n - 2):
            cur_idx += 1
            print("R Layer :", cur_idx)
            corresponding_L_idx = cur_idx - (2 * i + 3)
            corresponding_L_layer = hidden_layers[corresponding_L_idx]
            upper = f(previous_layer[: N // 2], previous_layer[N // 2:] + corresponding_L_layer[1: N: 2])
            lower = f(corresponding_L_layer[: N: 2], previous_layer[: N // 2]) + previous_layer[N // 2:]
            previous_layer = interlace(upper, lower)
            hidden_layers.append(previous_layer)

    # The last full L propagation. Need to do 1 more time.
    cur_idx += 1
    print("L Layer :", cur_idx)
    upper = f(x_LLR[: N: 2], x_LLR[1: N: 2] + previous_layer[N // 2:])
    lower = f(previous_layer[: N // 2], x_LLR[: N: 2]) + x_LLR[1: N: 2]
    previous_layer = concatenate(upper, lower)

    # This is for the Ex-BP decoder.
    upper_Ex = alpha[Ex_counter, : N // 2] * upper + beta[Ex_counter, N // 2:] * previous_layer[N // 2:]
    lower_Ex = alpha[Ex_counter, N // 2:] * lower + beta[Ex_counter, : N // 2] * previous_layer[: N // 2]
    Ex_counter += 1
    previous_layer = concatenate(upper_Ex, lower_Ex)

    hidden_layers.append(previous_layer)

    # Not the first L propagation layer.
    for i in range(n - 2):
        cur_idx += 1
        print("L Layer :", cur_idx)
        corresponding_R_idx = cur_idx - (2 * i + 3)
        corresponding_R_layer = hidden_layers[corresponding_R_idx]
        upper = f(previous_layer[: N: 2], previous_layer[1: N: 2] + corresponding_R_layer[N // 2:])
        lower = f(corresponding_R_layer[: N // 2], previous_layer[: N: 2]) + previous_layer[1: N: 2]
        previous_layer = concatenate(upper, lower)
        hidden_layers.append(previous_layer)

    # This is for the output layer.
    cur_idx += 1
    print("L Layer :", cur_idx)
    upper = f(previous_layer[: N: 2], previous_layer[1: N: 2] + R_LLR[N // 2:])
    lower = f(R_LLR[: N // 2], previous_layer[: N: 2]) + previous_layer[1: N: 2]
    previous_layer = concatenate(upper, lower)
    hidden_layers.append(previous_layer)
    #####################################################
    #####################################################

    # Take the last hidden layer as the output layer.
    output_layer = hidden_layers[-1]
    y_hat = float_to_bin(output_layer)

    # Return a length-N vector
    return y_hat


# Min-Sum Algorithm
def f(x, y):
    return tf.sign(x) * tf.sign(y) * tf.minimum(tf.abs(x), tf.abs(y))


def interlace(a, b):
    return tf.reshape(tf.stack([a, b], axis=-1), [-1])


def concatenate(a, b):
    return tf.concat([a, b], axis = 0)


def float_to_bin(a):
    return (tf.sign(a) + 1) / 2


def add_noise(signal):
    noise = np.random.normal(scale=sigma, size=(N, ))
    return signal + noise


if __name__ == '__main__':

    x_train = add_noise(np.zeros((N, )))
    y_train = np.zeros((N, ))

    # x is the input vector; y is the output vector (Length: N)
    x = tf.placeholder(tf.float64, shape=(N, ), name='x')
    y = tf.placeholder(tf.float64, shape=(N, ), name='y')

    # alpha and beta are the weights to be trained of the Ex-BP decoder.
    # Can also be initialized using "random_normal".
    alpha = tf.Variable(dtype=tf.float64, initial_value=tf.random_normal((iter_num, N, ), dtype=tf.float64),
                        expected_shape=(iter_num, N // 2, ), name='alpha', trainable=True)
    beta = tf.Variable(dtype=tf.float64, initial_value=tf.random_normal((iter_num, N, ), dtype=tf.float64),
                       expected_shape=(iter_num, N // 2, ), name='beta', trainable=True)

    # y_hat is a length-N vector.
    sess.run(tf.global_variables_initializer())
    x_test = add_noise(np.zeros((N, )))
    print(x_test)
    y_hat = sess.run(forwardprop(x_test, alpha, beta))
    print(sess.run(float_to_bin(x_test)))
    print(y_hat)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    update = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    alpha_val = sess.run(alpha)
    beta_val = sess.run(beta)
    print(alpha_val)
    print(beta_val)

    for epoch in range(100):
        print('Epoch: ', epoch)
        sess.run(update, feed_dict={x: x_train, y: y_train})
        print('cost :', sess.run(cost, feed_dict={x: x_train, y: y_train}))

    alpha_val = sess.run(alpha)
    beta_val = sess.run(beta)
    print(alpha_val)
    print(beta_val)

    # Play a sound when the process finishes.
    import winsound
    duration = 1000  # millisecond
    freq = 440       # Hz
    winsound.Beep(freq, duration)
