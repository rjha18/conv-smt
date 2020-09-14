import numpy as np
import tensorflow.compat.v1 as tf
# import matplotlib.pyplot as plt

tf.disable_v2_behavior()

# V and b are for display purposes here (filters will be displayed in V rows
# of b filters each)
V = 16
b = 16
K = V*b


stride = 8
k_sz = 12
sz = k_sz*k_sz


frames = np.load('./topography/small.npy')
frames = frames.reshape([-1, V, b])
print(frames.shape)
# N, H, W, C = frames.shape
N, H, W = frames.shape


batch_mode = True
batch_size = 32

# If you want to load learned features
load = False
load_pre = './'

if batch_mode:
    T = 1
    N_batches = 31
    load = True

    print(N)
    if N_batches * batch_size > N:
        print('Not enough data examples!')
        input()
else:
    T = 3

I_trajectory = tf.placeholder(tf.float32, shape=(batch_size, T, H, W, 1))
U = tf.placeholder(tf.float32, shape=(K, sz))

I = tf.reshape(I_trajectory, [-1, H, W, 1])


# Soft thresholding operator
def _soft_th(x, param):
    x_sign = tf.sign(x)
    x_reduced = tf.nn.relu(tf.subtract(tf.abs(x), param))
    x_soft = tf.multiply(x_sign, x_reduced)

    return x_soft


# Computes intitial size of feature maps
def compute_feats(I, filters):
    filters = tf.reshape(filters, [1, K, k_sz, k_sz])
    filters = tf.transpose(filters, [2, 3, 0, 1])
    alpha = tf.nn.conv2d(I, filters, strides=[
                         1, stride, stride, 1], padding='SAME')

    h1 = alpha.get_shape()[1]
    w1 = alpha.get_shape()[2]

    alpha = tf.reshape(alpha, [-1, h1, w1, K])

    return alpha, h1, w1


# Deconvolves features to reconstruct the input
def deconv_generator(alpha, filters):
    filters = tf.reshape(filters, [1, K, k_sz, k_sz])
    filters = tf.transpose(filters, [2, 3, 0, 1])
    img = tf.nn.conv2d_transpose(alpha, filters, output_shape=[
                                 batch_size*T, H, W, 1],
                                 strides=[1, stride, stride, 1],
                                 padding='SAME')
    return img


# MSE loss function
def MSE(y, y_hat):
    dim0 = tf.shape(y)[0]
    y = tf.reshape(y, [dim0, -1])
    y_hat = tf.reshape(y_hat, [dim0, -1])
    return tf.reduce_mean(tf.reduce_sum(tf.square(y-y_hat), axis=-1))


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


# reconstruct and compute loss
def rec_loss(alpha, filters, I):
    I_hat = deconv_generator(alpha, filters)
    return MSE(I, I_hat)


# Compute initial feature map and zero it out (you can also comment the *= 0
# line and see how it affects inference)
alpha, h1, w1 = compute_feats(I, U)
alpha *= 0

# eta is the gradient descent step size
# gamma is the sparsity penalty strength (large == more sparse)
eta = 2e-1
gamma = 4e-3

# These are functions used in the while loop. The step function corresponds to one iteration.
# The advantage is that TF is aware the loop exists and sets up the model faster.
loss_func = lambda alpha : rec_loss(alpha, U, I);
step_func = lambda prev,curr,y,t,hist : fista_loop(loss_func,prev,curr,y,t,hist,eta,gamma,rectify=True);
crit_func = lambda prev,curr,y,t,hist : tf.greater(1.0,0.0);

y = alpha
curr = alpha
prev = alpha

# hist keeps track of the loss for debug purposes
hist = tf.zeros((0, 1))

t = tf.constant(1.0)

[prev,curr,y,t,hist] = tf.while_loop(crit_func,step_func,[prev,curr,y,t,hist],\
    shape_invariants=[prev.get_shape(),curr.get_shape(),y.get_shape(),t.get_shape(),tf.TensorShape([None, 1])],back_prop=False,maximum_iterations=50);


alpha = tf.stop_gradient(curr)
I_hat = deconv_generator(alpha, U)
mse = MSE(I,I_hat);


# gradient descent step on features
grad_U = tf.gradients(xs=U, ys=mse)[0]
U_prime = U - 1e-0*grad_U


# keep track of everything on tensorboard
tf.summary.scalar('mse', mse)

tf.summary.image('I0', tf.reshape(
    tf.slice(I, [0, 0, 0, 0], [1, -1, -1, -1]), [-1, H, W, 1]))
tf.summary.image('I0_hat', tf.reshape(
    tf.slice(I_hat, [0, 0, 0, 0], [1, -1, -1, -1]), [-1, H, W, 1]))


# This parts displays the features nicely
leaves = tf.reshape(U, [V, b, sz])
tiled_leaves = tf.zeros((0, b*k_sz))

for i in range(V):

    row_leaves = tf.zeros((k_sz, 0))

    for j in range(b):
        row_leaves = tf.concat([row_leaves, tf.reshape(
            tf.slice(leaves, [i, j, 0], [1, 1, -1]), [k_sz, k_sz])], axis=-1)

    tiled_leaves = tf.concat([tiled_leaves, row_leaves], axis=0)

tf.summary.image('filters', tf.reshape(tiled_leaves, [1, V*k_sz, b*k_sz, 1]))
summary_op = tf.summary.merge_all()


# Project features to unit l2 norm after each gradient descent step
def project_basis(basis):
    norm = 1e-8+np.sqrt(np.sum(np.square(basis), axis=-1, keepdims=True))
    return basis/norm


summary_writer = tf.summary.FileWriter('./log/')
summary_writer.add_graph(tf.get_default_graph())


with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if load:
        F = np.load(load_pre+'bases.npy')
        F = F.reshape([-1, sz])
    else:
        F = project_basis(np.random.randn(K, sz))

    if batch_mode:

        sequence = frames[0:batch_size].reshape([batch_size, 1, H, W, 1])
        print(F.shape)
        [ALPHA] = sess.run([alpha], feed_dict={I_trajectory: sequence, U: F})

        alpha_shape = ALPHA.shape
        batch_alpha = np.zeros((N_batches*batch_size,)+alpha_shape[1:])

        for index in range(N_batches):
            print(index * batch_size)
            sequence = frames[index*batch_size:(index+1)*batch_size].reshape([batch_size, 1, H, W, 1])

            print('Example '+str(index)+'/'+str(N_batches))

            [ALPHA] = sess.run([alpha], feed_dict={I_trajectory: sequence, U: F})

            batch_alpha[index*batch_size:(index+1)*batch_size] = ALPHA

        np.save('batch_alpha.npy', batch_alpha)

    else:
        for epoch in range(100):
            for index in range(N):

                global_step = epoch*N+index

                fidx = np.random.randint(0, N - T, batch_size)
                sequence = np.zeros([batch_size,T,H,W,1]);
                sequence[:,[0],:,:,:] = frames[fidx].reshape([batch_size, 1, H, W, 1])
                
                # I am pretty sure there is a pythonic way of doing this more efficiently.
                
                for tidx in range(T-1):
                    sequence[:,[tidx+1],:,:,:] = frames[fidx+tidx+1].reshape([batch_size, 1, H, W, 1])
                

                summary_str, run_loss, F_prime, HIST = sess.run([summary_op, mse, U_prime, hist],\
                feed_dict={I_trajectory: sequence, U: F})

                # F_prime contains the updated features
                # If we do not project then we can arbitrarily improve the
                # loss by making magnitudes bigger and activations smaller:
                #
                # Code:  alpha_up = alpha/c
                # Feats: U_up = U*c
                #
                # alpha_up*U_up = (alpha/c)*U*c = alpha*U
                # which means the reconstruction is the same but alpha_up has
                # smaller l1 norm than alpha. Therefore we need to constraint
                # the magnitude of U to make this a well defined problem

                F = project_basis(F_prime)

                print("Loss ("+str(index)+"):", run_loss)

                # Uncomment these to plot the optimization history and see if
                # the optimization converges for your chosen eta and num of
                # iterations
                # plt.plot(HIST)
                # plt.show()

                summary_writer.add_summary(summary_str, global_step)

                # Save dictionary every 100 batches
                if global_step % 100 == 0:
                    np.save('bases.npy', F)
