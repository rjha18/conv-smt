import numpy as np
import tensorflow.compat.v1 as tf
from utils import infer_clusters, MSE, topographic_generator
import argparse
import os

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', nargs='?', const=1.0, type=float, default=1.0)
parser.add_argument('--gamma', nargs='?', const=1e-3, type=float, default=1e-3)
parser.add_argument('--epochs', nargs='?', const=100, type=int, default=100)
parser.add_argument('--verbosity', nargs='?', const=1, type=int, default=1)
parser.add_argument('--dir', nargs='?', const=False, type=bool, default=False)
args = parser.parse_args()
print(args)


# Hyperparameters
K_sqrt = 16                     # Number of filters in x or y direction
K = K_sqrt*K_sqrt               # Number of filters
k_sz = M = 12                   # Size of each filter in x or y direction
sz = k_sz*k_sz                  # Number of pixels in filter

T_sz = 5                        # Size of topological penalty map
T_stride = 2                    # Stride of topological penalty
sigma = args.sigma              # Smoothness of topological penalty
max_iters = 100                 # Maximum number of iterations

gamma = args.gamma              # Sparsity penalty
epochs = args.epochs            # number of epochs to train
batch_size = 512                # Number of batches
eta = 10e-0                     # Gradient Descent step size
result_dir = "./"               # Directory for the results
verbosity = args.verbosity      # Mod of iterations to print MSE

if args.dir:
    result_dir = "./Results/" + str(sigma) + "/" + str(gamma) + "/"
    os.makedirs(result_dir, exist_ok=True)


patches = np.load('small.npy')
N = patches.shape[0]

patches = patches.reshape([N, 12, 12])
patches = patches[:, :M, :M]
patches = patches.reshape([N, sz])


U = tf.placeholder(tf.float32, shape=(K, sz))

I = tf.placeholder(tf.float32, shape=(batch_size, M*M))

z, hist, loss = infer_clusters(I, U, gamma, eta, max_iters, batch_size,
                               K_sqrt, T_sz, T_stride, sigma)

I_hat = topographic_generator(z, U, batch_size, K_sqrt, T_sz, T_stride, sigma)
mse = MSE(I, I_hat)


# gradient descent step on features
grad_U = tf.gradients(xs=U, ys=mse)[0]
U_prime = U - 1e-0*grad_U

# keep track of everything on tensorboard
tf.summary.scalar('mse', mse)

tf.summary.image('I0', tf.reshape(
    tf.slice(I, [0, 0], [1, -1]), [-1, M, M, 1]))
tf.summary.image('I0_hat', tf.reshape(
    tf.slice(I_hat, [0, 0], [1, -1]), [-1, M, M, 1]))


# This parts displays the features nicely
leaves = tf.reshape(U, [K_sqrt, K_sqrt, 1, sz])

for plane in range(1):
    plane_leaves = tf.slice(leaves, [0, 0, plane, 0], [-1, -1, 1, -1])
    plane_leaves = tf.squeeze(plane_leaves, axis=2)
    tiled_leaves = tf.zeros((0, K_sqrt*k_sz))
    for i in range(K_sqrt):

        row_leaves = tf.zeros((k_sz, 0))

        for j in range(K_sqrt):
            row_leaves = tf.concat([row_leaves,
                                    tf.reshape(tf.slice(plane_leaves,
                                                        [i, j, 0],
                                                        [1, 1, -1]),
                                               [k_sz, k_sz])],
                                   axis=-1)

        tiled_leaves = tf.concat([tiled_leaves, row_leaves], axis=0)

    tf.summary.image('filters_'+str(plane),
                     tf.reshape(tiled_leaves,
                                [1, K_sqrt*k_sz, K_sqrt*k_sz, 1]))
summary_op = tf.summary.merge_all()


# Project features to unit l2 norm after each gradient descent step
def project_basis(basis):
    norm = 1e-8+np.sqrt(np.sum(np.square(basis), axis=-1, keepdims=True))
    return basis/norm


summary_writer = tf.summary.FileWriter(result_dir + 'log/')
summary_writer.add_graph(tf.get_default_graph())


with tf.Session() as sess:
    devices = sess.list_devices()
    print("Devices", devices)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    F = project_basis(np.random.randn(K, sz))

    for epoch in range(epochs):
        for index in range(N):

            global_step = epoch*N+index

            fidx = np.random.randint(0, N, batch_size)
            batch_I = patches[fidx].reshape([-1, M*M])

            summary_str, MSE, F_prime, HIST = sess.run([summary_op,
                                                        mse,
                                                        U_prime,
                                                        hist],
                                                       feed_dict={I: batch_I,
                                                                  U: F})

            F = project_basis(F_prime)

            if global_step % verbosity == 0:
                print("Loss ("+str(index)+"):", MSE)

            summary_writer.add_summary(summary_str, global_step)

            # Save dictionary every 100 batches
            if global_step % 100 == 0:
                np.save(result_dir + 'bases.npy', F)
