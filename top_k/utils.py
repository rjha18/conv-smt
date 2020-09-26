import tensorflow as tf
import numpy as np


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


def gd_loop(loss_func, y, hist, eta):
    loss = loss_func(y)
    hist = tf.concat((hist, tf.reshape(loss, [1, 1])), axis=0)

    grad_y = tf.gradients(xs=y, ys=loss)[0]

    y = tf.nn.relu(y-eta*grad_y)

    return [y, hist]


# MSE loss function
def MSE(y, y_hat):
    dim0 = tf.shape(y)[0]
    y = tf.reshape(y, [dim0, -1])
    y_hat = tf.reshape(y_hat, [dim0, -1])
    return tf.reduce_mean(tf.reduce_sum(tf.square(y-y_hat), axis=-1))


def gaussian_box_3d(k_sz):
    W = (k_sz-1)/2.0
    seq = np.linspace(-W, W, k_sz)

    xx, yy, zz = np.meshgrid(seq, seq, seq)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2))
    kernel /= np.sum(kernel)

    return tf.constant(np.float32(kernel))


def gaussian_box_2d(k_sz, sigma=1.0):
    W = (k_sz-1)/2.0
    seq = np.linspace(-W, W, k_sz)

    xx, yy = np.meshgrid(seq, seq)
    kernel = np.exp(-(xx**2 + yy**2)/np.square(sigma))
    kernel /= np.sum(kernel)

    return tf.constant(np.float32(kernel))


def topographic_penalty3D(feats, k_sz, K_sqrt):
    gaussian_box = gaussian_box_3d(k_sz)
    gaussian_box = tf.reshape(gaussian_box, [k_sz, k_sz, k_sz, 1, 1])

    feats = tf.reshape(feats, [-1, K_sqrt, K_sqrt, K_sqrt, 1])
    penalty = tf.nn.conv3d(tf.square(feats),
                           gaussian_box,
                           strides=[1, 1, 1, 1, 1],
                           padding='SAME')

    penalty = tf.sqrt(penalty+1e-6)
    penalty = tf.reduce_sum(penalty)

    return penalty


def abs_normalize(z):
    print(z)
    z_abs = np.abs(z)
    print(np.max(z_abs) - z_abs)
    return np.max(z_abs) - z_abs


def topographic_generator(z, U, batch_size, K_sqrt, T_sz, T_stride, sigma):
    K = K_sqrt*K_sqrt
    # gaussian_box = gaussian_box_2d(T_sz, sigma)
    # gaussian_box = tf.reshape(gaussian_box, [T_sz, T_sz, 1, 1])
    w = tf.nn.conv2d_transpose(abs_normalize(z),
                               tf.constant(1.0, shape=[T_sz, T_sz, 1, 1]),
                               output_shape=[batch_size, K_sqrt, K_sqrt, 1],
                               strides=[1, T_stride, T_stride, 1],
                               padding='VALID')
    w = tf.reshape(w, [batch_size, K])

    I_hat = tf.matmul(w, U)
    return I_hat


def topographic_penalty2D(feats, k_sz, K_sqrt):
    gaussian_box = gaussian_box_2d(k_sz)
    gaussian_box = tf.reshape(gaussian_box, [k_sz, k_sz, 1, 1])

    feats = tf.reshape(feats, [-1, K_sqrt, K_sqrt, 1])
    penalty = tf.nn.conv2d(tf.square(feats),
                           gaussian_box,
                           strides=[1, 1, 1, 1],
                           padding='SAME')

    penalty = tf.sqrt(penalty+1e-6)
    penalty = tf.reduce_sum(penalty)

    return penalty


# def infer_topographic_sparse_code(I, U, gamma, eta, max_iters, k_sz, K_sqrt):
#     gen_func = lambda y: topographic_generator(y)
#     loss_func = lambda y: MSE(I, gen_func(y))
#     step_func = lambda y, hist: fista_loop(loss_func, y, hist, eta)
#     crit_func = lambda y, hist: tf.greater(1.0, 0.0)

#     r_init = tf.zeros_like(tf.matmul(I, U, transpose_b=True))
#     hist = tf.zeros((0, 1))

#     y = r_init

#     [y, hist] = tf.while_loop(crit_func, step_func, [y, hist],
#                               shape_invariants=[y.get_shape(),
#                                                 tf.TensorShape([None,  1])],
#                               back_prop=False,
#                               maximum_iterations=max_iters)

#     r = tf.stop_gradient(y)
#     loss = loss_func(r)

#     return r, hist, loss


def infer_topographic_sparse_code(I, U, gamma, eta, max_iters, k_sz, K_sqrt):
    gen_func = lambda y: tf.matmul(y, U)
    loss_func = lambda y: MSE(I, gen_func(y)) \
        + gamma*topographic_penalty2D(y, k_sz, K_sqrt)
    step_func = lambda y, hist: gd_loop(loss_func, y, hist, eta)
    crit_func = lambda y, hist: tf.greater(1.0, 0.0)

    r_init = tf.zeros_like(tf.matmul(I, U, transpose_b=True))
    hist = tf.zeros((0, 1))

    y = r_init

    [y, hist] = tf.while_loop(crit_func, step_func, [y, hist],
                              shape_invariants=[y.get_shape(),
                                                tf.TensorShape([None, 1])],
                              back_prop=False,
                              maximum_iterations=max_iters)

    r = tf.stop_gradient(y)
    loss = loss_func(r)

    return r, hist, loss


def infer_clusters(I, U, gamma, eta, max_iters, batch_size, K_sqrt, T_sz,
                   T_stride, sigma):
    gen_func = lambda y: topographic_generator(y, U, batch_size, K_sqrt, T_sz,
                                               T_stride, sigma)
    loss_func = lambda y: tf.reduce_mean(tf.reduce_sum(tf.square(I-gen_func(y)), axis=-1))
    step_func = lambda prev, curr, y, t, hist: fista_loop(loss_func, prev, curr, y, t, hist, eta, gamma, rectify=False)
    crit_func = lambda prev, curr, y, t, hist: tf.greater(1.0, 0.0)

    # Sigma_U = tf.matmul(U,U,transpose_b=True)

    h1 = np.int32((K_sqrt-T_sz)/T_stride + 1)
    w1 = np.int32((K_sqrt-T_sz)/T_stride + 1)
    z_init = tf.zeros((batch_size, h1, w1, 1))

    hist = tf.zeros((0, 1))

    prev = z_init
    curr = z_init
    y = z_init

    t = tf.constant(1.0)

    [prev, curr, y, t, hist] = tf.while_loop(crit_func,
                                             step_func,
                                             [prev, curr, y, t, hist],
                                             shape_invariants=[
                                                prev.get_shape(),
                                                curr.get_shape(),
                                                y.get_shape(),
                                                t.get_shape(),
                                                tf.TensorShape([None, 1])],
                                             back_prop=False,
                                             maximum_iterations=max_iters)

    z = tf.stop_gradient(curr)
    loss = loss_func(z)

    return z, hist, loss


def infer_sparse_code(I, U, M, gamma, eta, max_iters):
    dot_products = tf.matmul(I, tf.tranpose(U))
    idx = tf.nn.top_k(dot_products)
    M = tf.zeros((N, K))
    M <- idx
    gen_func = lambda y : tf.matmul(M*y, U)
    loss_func = lambda y: tf.reduce_mean(tf.reduce_sum(tf.square(I-gen_func(y)), axis=-1))
    step_func = lambda y, hist: gd_loop(loss_func, y, hist, eta)
    crit_func = lambda prev, curr, y, t, hist: tf.greater(1.0, 0.0)

    # Sigma_U = tf.matmul(U,U,transpose_b=True)

    r_init = tf.zeros_like(tf.matmul(I, U, transpose_b=True))
    # r_init = tf.matmul(r_init,tf.linalg.inv(Sigma_U),transpose_b=True);
    hist = tf.zeros((0, 1))

    prev = r_init
    curr = r_init
    y = r_init

    t = tf.constant(1.0)

    [prev, curr, y, t, hist] = tf.while_loop(crit_func,
                                             step_func,
                                             [prev, curr, y, t, hist],
                                             shape_invariants=[
                                                prev.get_shape(),
                                                curr.get_shape(),
                                                y.get_shape(),
                                                t.get_shape(),
                                                tf.TensorShape([None, 1])],
                                             back_prop=False,
                                             maximum_iterations=max_iters)

    r = tf.stop_gradient(curr)
    loss = loss_func(r)
    return r, hist, loss
