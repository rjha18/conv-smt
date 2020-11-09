import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from transformer import SPN

tf.disable_v2_behavior()


H_LINE = np.zeros((1, 7, 7))
V_LINE = np.zeros((1, 7, 7))

H_LINE[:, 2:4, :] = 1.0
V_LINE[:, :, 2:4] = 1.0

h_line = tf.constant(np.float32(H_LINE))
v_line = tf.constant(np.float32(V_LINE))
U = tf.concat([h_line, v_line], axis=0)

THETA = np.zeros((2, 2, 6))


TX = 1.0
TY = 1.0
R = -1.0
SX = 0.0
SY = 0.0
Z = 0.0

THETA[0, 0, 0] = TX
THETA[0, 0, 1] = TY
THETA[0, 0, 2] = R
THETA[0, 0, 3] = SX
THETA[0, 0, 4] = SY
THETA[0, 0, 5] = Z

THETA[0, 1, 0] = -TX
THETA[0, 1, 1] = 0.0
THETA[0, 1, 2] = R
THETA[0, 1, 3] = SX
THETA[0, 1, 4] = SY
THETA[0, 1, 5] = Z


THETA[1, 0, 0] = 0.0
THETA[1, 0, 1] = -TY
THETA[1, 0, 2] = R
THETA[1, 0, 3] = SX
THETA[1, 0, 4] = SY
THETA[1, 0, 5] = -Z

THETA[1, 1, 0] = TX
THETA[1, 1, 1] = 0.0
THETA[1, 1, 2] = R
THETA[1, 1, 3] = SX
THETA[1, 1, 4] = SY
THETA[1, 1, 5] = -Z

THETA = THETA.reshape([-1, 6])

theta = tf.constant(THETA)

U = tf.tile(tf.reshape(U, [2, 7, 7, 1]), [2, 1, 1, 1])

# TU = transformer(U, theta)

print(theta.shape)
TU, tgrids, grids, mats = SPN(U, theta, 7, 7)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    [TLINES, TGRIDS, GRIDS, MATS] = sess.run([TU, tgrids, grids, mats],
                                             feed_dict={})

    TGRIDS = np.transpose(TGRIDS, [0, 2, 3, 1])
    TGRIDS = TGRIDS.reshape([-1, 49, 2])

    for i in range(4):
        plt.scatter(GRIDS[:, 0], GRIDS[:, 1], color='green')
        plt.scatter(TGRIDS[i, :, 0], TGRIDS[i, :, 1], color='red')
        plt.savefig('img/1.png')

    plt.subplot(1, 2, 1)
    plt.imshow(H_LINE.reshape([7, 7]))
    plt.subplot(1, 2, 2)
    plt.imshow(TLINES[0].reshape([7, 7]))
    plt.show()
    plt.savefig('img/2.png')

    plt.subplot(1, 2, 1)
    plt.imshow(V_LINE.reshape([7, 7]))
    plt.subplot(1, 2, 2)
    plt.imshow(TLINES[1].reshape([7, 7]))
    plt.show()
    plt.savefig('img/3.png')

    plt.subplot(1, 2, 1)
    plt.imshow(H_LINE.reshape([7, 7]))
    plt.subplot(1, 2, 2)
    plt.imshow(TLINES[2].reshape([7, 7]))
    plt.show()
    plt.savefig('img/4.png')

    plt.subplot(1, 2, 1)
    plt.imshow(V_LINE.reshape([7, 7]))
    plt.subplot(1, 2, 2)
    plt.imshow(TLINES[3].reshape([7, 7]))
    plt.show()
    plt.savefig('img/5.png')
