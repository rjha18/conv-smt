import tensorflow as tf
import numpy as np
from transformer import SPN


def generate_grid(scale, resolution, dims):
    x_C = np.linspace(-scale, scale, resolution)

    axes = []

    for dim in range(dims):
        axes += [x_C]

    packed_grid = np.meshgrid(*axes)

    np_grid = packed_grid[0].reshape([-1, 1])

    for dim in range(dims-1):
        np_grid = np.concatenate([np_grid, packed_grid[dim+1]
                                 .reshape([-1, 1])],
                                 axis=1)

    grid = tf.constant(np.float32(np_grid))
    # grid = tf.get_variable("grid",initializer=np.float32(np_grid))

    return grid


def pairwise_dist(A, B):
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    D = tf.sqrt(1e-12+tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb,
                                 0.0))

    return D


def _soft_th(x, param):
    x_sign = tf.sign(x)
    x_reduced = tf.nn.relu(tf.subtract(tf.abs(x), param))
    x_soft = tf.multiply(x_sign, x_reduced)

    return x_soft


# FISTA loop for sparse inference: rectify=True makes the code nonnegative
def fista_loop(loss_func, prev, curr, y, t, hist, eta, gamma, rectify=False):
    loss = loss_func(y)
    hist = tf.concat([hist, tf.reshape(loss, [1, 1])], axis=0)

    grad_y = tf.gradients(xs=y, ys=loss)[0]

    y = y - eta*grad_y
    y = tf.nn.relu(_soft_th(y, eta*gamma))

    if rectify:
        y = tf.nn.relu(y)
    y = tf.stop_gradient(y)

    prev = curr
    curr = y
    y = curr + (((t+1)-2)/((t+1)+1))*(curr-prev)

    y = tf.stop_gradient(y)

    t += 1.0

    return [prev, curr, y, t, hist]


def gd_loop(loss_func, y, theta, hist, eta):
    loss = loss_func(y, theta)
    hist = tf.concat((hist, tf.reshape(loss, [1, 1])), axis=0)

    grad_y = tf.gradients(xs=y, ys=loss)[0]
    y = tf.nn.relu(y-eta*grad_y)

    grad_theta = tf.gradients(xs=theta, ys=loss)[0]
    theta = theta-eta*grad_theta

    return [y, theta, hist]


# MSE loss function
def MSE(y, y_hat):
    dim0 = tf.shape(y)[0]
    y = tf.reshape(y, [dim0, -1])
    y_hat = tf.reshape(y_hat, [dim0, -1])
    return tf.reduce_mean(tf.reduce_sum(tf.square(y-y_hat), axis=-1))


# Generate a mask for the filters
def create_mask(I, U, batch_size, k, K):
    # find the k closest filters to the image
    dot_products = tf.matmul(I, tf.transpose(U))
    vals, idx = tf.nn.top_k(dot_products, k)

    # Get the coordinates of the top_k filters for each item in batch
    rows = tf.range(0, batch_size, 1)
    rows = tf.tile(rows, [k])
    rows = tf.reshape(rows, [k, -1])
    rows = tf.transpose(rows)
    rows = tf.stack([rows, idx], axis=2)

    # Reshape for scatter
    indices = tf.reshape(rows, [-1, 2])

    # Generate mask of 0s and 1s where 1s denote a top_k filter
    mask = tf.scatter_nd(indices,
                         tf.ones([indices.shape[0]]),
                         tf.constant([batch_size, K]))
    return mask


def infer_sparse_code(I, U, M, gamma, eta, max_iters):
    theta = tf.zeros((I.shape[0] * U.shape[0], 6))
    print(theta.shape)
    U_ext = tf.reshape(U, [1, U.shape[0], 144])
    U_ext = tf.tile(U_ext, [I.shape[0], 1, 1])
    U_ext = tf.reshape(U_ext, [-1, 12, 12, 1])
    gen_func = lambda y, theta: tf.matmul(M*y, tf.reshape(SPN(U_ext, theta, 12, 12), [I.shape[0], -1, 144]))
    loss_func = lambda y, theta: tf.reduce_mean(tf.reduce_sum(tf.square(I-gen_func(y, theta)), axis=-1))
    step_func = lambda y, theta, hist: gd_loop(loss_func, y, theta, hist, eta)
    crit_func = lambda y, theta, hist: tf.greater(1.0, 0.0)

    # Sigma_U = tf.matmul(U,U,transpose_b=True)

    r_init = tf.zeros_like(tf.matmul(I, U, transpose_b=True))
    # r_init = tf.matmul(r_init,tf.linalg.inv(Sigma_U),transpose_b=True);
    hist = tf.zeros((0, 1))

    y = r_init

    theta = tf.random.normal((tf.shape(I)[0] * tf.shape(U)[0], 6))

    [y, theta, hist] = tf.while_loop(crit_func,
                              step_func,
                              [y, theta, hist],
                              shape_invariants=[
                                  y.get_shape(),
                                  theta.get_shape(),
                                  tf.TensorShape([None, 1])],
                              back_prop=False,
                              maximum_iterations=max_iters)

    r = tf.stop_gradient(y)
    loss = loss_func(r, theta)
    return r, hist, loss, theta
