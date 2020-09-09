import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()


from utils import infer_topographic_sparse_code,MSE;

# V and b are for display purposes here (filters will be displayed in V rows
# of b filters each)
K_sqrt = 16;
K = K_sqrt*K_sqrt
T_sz = 5;

stride = 12
k_sz = 12
sz = k_sz*k_sz


patches = np.load('small.npy');
N = patches.shape[0];

M = k_sz = 12;
sz = M*M;

patches = patches.reshape([N,12,12]);
patches = patches[:,:M,:M];
patches = patches.reshape([N,sz]);




batch_size = 512;



I = tf.placeholder(tf.float32, shape=(batch_size,M*M))
U = tf.placeholder(tf.float32, shape=(K, sz))

gamma = 1e-3;
eta = 10e-0;
max_iters = 100;
r,hist,loss = infer_topographic_sparse_code(I,U,gamma,eta,max_iters,T_sz,K_sqrt)

I_hat = tf.matmul(r,U);
mse = MSE(I,I_hat);


# gradient descent step on features
grad_U = tf.gradients(xs=U, ys=mse)[0]
U_prime = U - 1e-0*grad_U

# keep track of everything on tensorboard
tf.summary.scalar('mse', mse)

tf.summary.image('I0', tf.reshape(
	tf.slice(I,[0,0],[1,-1]),[-1,M,M,1]))
tf.summary.image('I0_hat', tf.reshape(
	tf.slice(I_hat,[0,0],[1,-1]),[-1,M,M,1]))


# This parts displays the features nicely
leaves = tf.reshape(U, [K_sqrt, K_sqrt, 1, sz])

for plane in range(1):
	plane_leaves = tf.slice(leaves,[0,0,plane,0],[-1,-1,1,-1]);
	plane_leaves = tf.squeeze(plane_leaves,axis=2);
	tiled_leaves = tf.zeros((0, K_sqrt*k_sz))
	for i in range(K_sqrt):

		row_leaves = tf.zeros((k_sz, 0))

		for j in range(K_sqrt):
			row_leaves = tf.concat([row_leaves, tf.reshape(
				tf.slice(plane_leaves, [i, j, 0], [1, 1, -1]), [k_sz, k_sz])], axis=-1)

		tiled_leaves = tf.concat([tiled_leaves, row_leaves], axis=0)

	tf.summary.image('filters_'+str(plane), tf.reshape(tiled_leaves, [1, K_sqrt*k_sz, K_sqrt*k_sz, 1]))
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


	F = project_basis(np.random.randn(K, sz))

	for epoch in range(100):
		for index in range(N):

			global_step = epoch*N+index

			fidx = np.random.randint(0,N,batch_size)
			batch_I = patches[fidx].reshape([-1,M*M])
			
			
			summary_str, MSE, F_prime, HIST = sess.run([summary_op, mse, U_prime, hist],\
			feed_dict={I: batch_I, U: F})

			F = project_basis(F_prime)

			print("Loss ("+str(index)+"):", MSE)

			#plt.plot(HIST)
			#plt.show()


			summary_writer.add_summary(summary_str, global_step)

			# Save dictionary every 100 batches
			if global_step % 100 == 0:
				np.save('bases.npy', F)
			
				
				
				
				
				
				
				
				
