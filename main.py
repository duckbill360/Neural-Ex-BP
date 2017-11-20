# Neural Ex-BP

import polar_codes
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10000)
np.core.arrayprint._line_width = 10000

############### Polar Codes PARAMETERS ###############
N_codewords = 24000
BATCH_SIZE = 120
N_epochs = 100
learning_rate = 0.0002
iter_num = 5
N = 1024   # code length
n = int(np.log2(N))
layers_per_iter = 2 * n - 2     # Need an additional L layer.
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 2.0
######################################################

Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10.0))
sigma = pow(Var, 1 / 2)


def generate_interleaver_matrix(N):
    matrix = np.zeros(shape=(N, N))
    upper, lower = 0, N // 2

    for i in range(N):
        if i % 2 == 0:
            matrix[upper, i] = 1
            upper += 1
        elif i % 2 == 1:
            matrix[lower, i] = 1
            lower += 1

    return matrix


B_N = polar_codes.permutation_matrix(N)
G = polar_codes.generate_G_N(N)
interleaver = generate_interleaver_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
BIG_NUM = 1000000

frozen_list = np.ones((1, N))

for index in frozen_indexes:
    frozen_list[0, index] = 0

inverse_frozen_list = np.ones((1, N)) - frozen_list

frozen_list = np.dot(frozen_list, B_N)
inverse_frozen_list = np.dot(inverse_frozen_list, B_N)


def forwardprop(x, alpha, beta):

    channel_LLR = 2 * x / np.power(sigma, 2)
    print("channel_LLR's shape :", channel_LLR.shape)

    # The list of all hidden layers
    hidden_layers = []

    # Initialize decision_LLR
    # LLR = [0 for i in range(N)]
    LLR = np.zeros((BATCH_SIZE, N), dtype=np.float64)

    for i in frozen_indexes:
        LLR[:, i] = BIG_NUM

    LLR_permuted = np.dot(LLR, B_N)
    # This must be checked.

    decision_LLR = tf.constant(LLR_permuted, dtype=tf.float64)
    print('decision ', decision_LLR.shape)
    N_2_zeros = tf.zeros(shape=(BATCH_SIZE, N // 2), dtype=tf.float64)
    Ex_counter = 0

    #####################################################
    ############## The first R Propagation ##############
    # The first R hidden layer, so take decision_LLR as an input, and view L as 0
    cur_idx = 0
    print('R Layer :', cur_idx)
    upper = f(decision_LLR[:, : N // 2], decision_LLR[:, N // 2:] + N_2_zeros)
    lower = f(N_2_zeros, decision_LLR[:, : N // 2]) + decision_LLR[:, N // 2:]
    previous_layer = interlace(upper, lower)
    hidden_layers.append(previous_layer)

    # Not the first R hidden layer, so take the previous layers as inputs and view L as 0
    for i in range(n - 2):
        cur_idx += 1
        print("R Layer :", cur_idx)
        upper = f(previous_layer[:, : N // 2], previous_layer[:, N // 2:] + N_2_zeros)
        lower = f(N_2_zeros, previous_layer[:, : N // 2]) + previous_layer[:, N // 2:]
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
        upper = f(channel_LLR[:, : N: 2], channel_LLR[:, 1: N: 2] + previous_layer[:, N // 2:])
        lower = f(previous_layer[:, : N // 2], channel_LLR[:, : N: 2]) + channel_LLR[:, 1: N: 2]
        # previous_layer = concatenate(upper, lower)

        # This is for the Ex-BP decoder.
        upper_Ex = tf.multiply(alpha[Ex_counter][0, : N // 2], upper) + \
                   tf.multiply(beta[Ex_counter][0, N // 2:], previous_layer[:, N // 2:])
        lower_Ex = tf.multiply(alpha[Ex_counter][0, N // 2:], lower) + \
                   tf.multiply(beta[Ex_counter][0, : N // 2], previous_layer[:, : N // 2])
        # upper_Ex = alpha[Ex_counter][: N // 2] * upper + beta[Ex_counter][N // 2:] * previous_layer[:, N // 2:]
        # lower_Ex = alpha[Ex_counter][N // 2:] * lower + beta[Ex_counter][: N // 2] * previous_layer[:, : N // 2]
        Ex_counter += 1
        previous_layer = concatenate(upper_Ex, lower_Ex)
        ###############################

        hidden_layers.append(previous_layer)

        # Not the first L propagation layer.
        for i in range(n - 2):
            cur_idx += 1
            print("L Layer :", cur_idx)
            corresponding_R_idx = cur_idx - (2 * i + 3)
            corresponding_R_layer = hidden_layers[corresponding_R_idx]
            upper = f(previous_layer[:, : N: 2], previous_layer[:, 1: N: 2] + corresponding_R_layer[:, N // 2:])
            lower = f(corresponding_R_layer[:, : N // 2], previous_layer[:, : N: 2]) + previous_layer[:, 1: N: 2]
            previous_layer = concatenate(upper, lower)
            hidden_layers.append(previous_layer)


        ###### R Propagation
        # The first R propagation layer.
        cur_idx += 1
        print("R Layer :", cur_idx)
        upper = f(decision_LLR[:, : N // 2], decision_LLR[:, N // 2:] + previous_layer[:, 1: N: 2])
        lower = f(previous_layer[:, : N: 2], decision_LLR[:, : N // 2]) + decision_LLR[:, N // 2:]
        previous_layer = interlace(upper, lower)
        hidden_layers.append(previous_layer)

        # Not the first R propagation layer.
        for i in range(n - 2):
            cur_idx += 1
            print("R Layer :", cur_idx)
            corresponding_L_idx = cur_idx - (2 * i + 3)
            corresponding_L_layer = hidden_layers[corresponding_L_idx]
            upper = f(previous_layer[:, : N // 2], previous_layer[:, N // 2:] + corresponding_L_layer[:, 1: N: 2])
            lower = f(corresponding_L_layer[:, : N: 2], previous_layer[:, : N // 2]) + previous_layer[:, N // 2:]
            previous_layer = interlace(upper, lower)
            hidden_layers.append(previous_layer)

    # The last full L propagation. Need to do 1 more time.
    cur_idx += 1
    print("L Layer :", cur_idx)
    upper = f(channel_LLR[:, : N: 2], channel_LLR[:, 1: N: 2] + previous_layer[:, N // 2:])
    lower = f(previous_layer[:, : N // 2], channel_LLR[:, : N: 2]) + channel_LLR[:, 1: N: 2]
    # previous_layer = concatenate(upper, lower)

    # This is for the Ex-BP decoder.
    upper_Ex = tf.multiply(alpha[Ex_counter][0, : N // 2], upper) + tf.multiply(beta[Ex_counter][0, N // 2:], previous_layer[:, N // 2:])
    lower_Ex = tf.multiply(alpha[Ex_counter][0, N // 2:], lower) + tf.multiply(beta[Ex_counter][0, : N // 2], previous_layer[:, : N // 2])
    Ex_counter += 1
    previous_layer = concatenate(upper_Ex, lower_Ex)
    ###############################
    hidden_layers.append(previous_layer)

    # Not the first L propagation layer.
    for i in range(n - 2):
        cur_idx += 1
        print("L Layer :", cur_idx)
        corresponding_R_idx = cur_idx - (2 * i + 3)
        corresponding_R_layer = hidden_layers[corresponding_R_idx]
        upper = f(previous_layer[:, : N: 2], previous_layer[:, 1: N: 2] + corresponding_R_layer[:, N // 2:])
        lower = f(corresponding_R_layer[:, : N // 2], previous_layer[:, : N: 2]) + previous_layer[:, 1: N: 2]
        previous_layer = concatenate(upper, lower)
        hidden_layers.append(previous_layer)

    # This is for the output layer.
    cur_idx += 1
    print("L Layer :", cur_idx)
    upper = f(previous_layer[:, : N: 2], previous_layer[:, 1: N: 2] + decision_LLR[:, N // 2:])
    lower = f(decision_LLR[:, : N // 2], previous_layer[:, : N: 2]) + previous_layer[:, 1: N: 2]
    previous_layer = concatenate(upper, lower)
    hidden_layers.append(previous_layer)
    #####################################################
    #####################################################

    # Take the last hidden layer as the output layer.
    output_layer = hidden_layers[-1]
    # output_layer = output_layer * frozen_list + output_layer * inverse_frozen_list * BIG_NUM

    y_hat = tf.negative(output_layer)

    # Return a length-N vector
    return y_hat


# Min-Sum Algorithm
def f(x, y):
    sign = tf.multiply(tf.sign(x), tf.sign(y))
    minimum = tf.minimum(tf.abs(x), tf.abs(y))
    return tf.multiply(sign, minimum)


def interlace(a, b):
    output = tf.concat([a, b], axis=1)
    return tf.matmul(output, interleaver)


def concatenate(a, b):
    return tf.concat([a, b], axis=1)


def add_noise(signal):
    noise = np.random.normal(scale=sigma, size=signal.shape)
    return signal + noise


if __name__ == '__main__':

    import timeit
    start = timeit.default_timer()

    ###### Training Data ######
    x_train = add_noise(np.ones((N_codewords, N), dtype=np.float64))
    y_train = np.zeros((N_codewords, N), dtype=np.float64)


    ###### Variable Initialization ######
    x = tf.placeholder(dtype=tf.float64, shape=(BATCH_SIZE, N), name='x')
    y = tf.placeholder(dtype=tf.float64, shape=(BATCH_SIZE, N), name='y')


    # alpha and beta are the weights to be trained of the Ex-BP decoder.
    alpha = [tf.Variable(dtype=tf.float64, initial_value=tf.ones((1, N), dtype=tf.float64), name='alpha', trainable=True)
             for _ in range(iter_num)]
    beta = [tf.Variable(dtype=tf.float64, initial_value=tf.random_normal((1, N), dtype=tf.float64), name='beta', trainable=True)
            for _ in range(iter_num)]


    ###### y_hat is a length-N vector. ######
    y_hat = forwardprop(x, alpha, beta)


    ###### Cost Function & Training Operation ######
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat))
    # cost = tf.reduce_sum(-tf.log(1 - y_hat))      # hand-coded cost function
    # cost = tf.reduce_sum(tf.square(y - y_hat))
    # cost = tf.losses.mean_squared_error(labels=y, predictions=y_hat)

    update = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    ###### Training ######
    with tf.Session() as sess:

        error_lst = []
        sess.run(tf.global_variables_initializer())

        alpha_val = sess.run(alpha)
        beta_val = sess.run(beta)
        print('Initial alpha:', alpha_val)
        print('Initial beta:', beta_val)

        # Run through N_epochs times.
        for epoch in range(N_epochs):
            total_err = 0
            print('Epoch: ', epoch)
            for i in range(N_codewords // BATCH_SIZE):
                offset = i * BATCH_SIZE
                err, _ = sess.run([cost, update], feed_dict={x: x_train[offset:offset + BATCH_SIZE, :],
                                                             y: y_train[offset:offset + BATCH_SIZE, :]})
                total_err += err
                print('cost ', offset, ':', err)
            error_lst.append(total_err)

        print('Error:', error_lst, '\n\n')
        error_lst = np.array(error_lst) / max(error_lst)
        plt.plot(error_lst)
        plt.show()

        alpha_val = sess.run(alpha)
        beta_val = sess.run(beta)
        print('Trained alpha:', alpha_val)
        print('Trained beta:', beta_val)

        # Save the weight for validation.
        np.save('alpha.npy', alpha_val)
        np.save('beta.npy', beta_val)
        print('Parameters saved.')

        # writer = tf.summary.FileWriter("TensorBoard/", graph=sess.graph)
        # TensorBoard command: tensorboard --logdir="./TensorBoard"

    stop = timeit.default_timer()
    print('\nRun time :', (stop - start) // 60, 'minutes,', (stop - start) % 60, 'seconds')

    # Play a sound when the process finishes.
    import winsound
    duration = 1000  # millisecond
    freq = 440       # Hz
    winsound.Beep(freq, duration)
