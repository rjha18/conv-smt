

import os.path;
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np;
import matplotlib.pyplot as plt;



class data_source:

	def __init__(self,kernel_size=7,sigma=1.0):
		self.kernel_size = kernel_size;
		self.border = np.int32(np.floor((self.kernel_size-1)/2));
		self.sigma = sigma;		
	
	def build_fnm(self,prefix,n):
		return prefix+'frame'+"{0:0=4d}".format(n)+'.png';
		
	
	def frames_to_npy(self,input_dir,output_fnm,start=1,subsample=1,pre_h=0,suff_h=0,pre_w=0,suff_w=0,frames_threshold=10000):
		
		frame_fnm = self.build_fnm(input_dir,start);
		print(frame_fnm)
		
		if not os.path.isfile(frame_fnm):
			print("No frames exist in directory!")
			return;
			
		first_frame = plt.imread(frame_fnm);
		
		frame_shape = first_frame.shape;
		extended_height = frame_shape[0] - (pre_h+suff_h);
		extended_width = frame_shape[1] - (pre_w+suff_w);
		
		
		N = start;
	
		while os.path.isfile(frame_fnm):
			N += 1;
			frame_fnm = self.build_fnm(input_dir,N);
		N -= 1;
		
		N = np.min([N,frames_threshold]);

		frames = np.zeros((N,extended_height,extended_width,1));
		
		for n in range(start,N):
			frame_fnm = self.build_fnm(input_dir,n+1);
			frame = plt.imread(frame_fnm);
			frame = frame[pre_h:(frame.shape[0]-suff_h),pre_w:(frame.shape[1]-suff_w)];
			frame = frame.reshape([1,extended_height,extended_width,1]);
			frames[n] = frame;
			
		frames = frames[::subsample];
		np.save(output_fnm,frames);
		
		print('Processed video:',output_fnm)
		print(frames)
	
	
	
	def contrast_normalize(self,I):
	
		dist = tf.distributions.Normal(loc=0., scale=self.sigma)

		W = (self.kernel_size-1)/2.0;
		box_x = tf.lin_space(-W,W,self.kernel_size);
		box_y = tf.lin_space(-W,W,self.kernel_size);
		
		prob_x = dist.prob(box_x);
		prob_y = dist.prob(box_y);
		
		gaussian_box = tf.matmul(tf.reshape(prob_x,[self.kernel_size,1]),tf.reshape(prob_y,[1,self.kernel_size]));
		gaussian_box = tf.reshape(gaussian_box,[self.kernel_size,self.kernel_size,1,1]);
		gaussian_box = tf.divide(gaussian_box,tf.reduce_sum(gaussian_box));

		avg_I = tf.nn.conv2d(I,gaussian_box,strides=[1,1,1,1],padding='SAME');
		normalized_I = I - avg_I;
		
		img_H = tf.shape(I)[1];
		img_W = tf.shape(I)[2];
		
		normalized_I = tf.slice(normalized_I,[0,self.border,self.border,0],[-1,img_H-2*self.border,img_W-2*self.border,-1]);

		return [normalized_I];
	
	def process_npy(self,input_fnm,output_fnm):
		
		vid = np.load(input_fnm);
		N,extended_height,extended_width,_ = vid.shape;
		
		
		self.H = H = extended_height - 2*self.border;
		self.W = W = extended_width - 2*self.border;


		I = tf.placeholder(tf.float32,[None,extended_height,extended_width,1]);
		[normalized_I] = self.contrast_normalize(I);
		
		batch_size = 32;
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			

			processed_vid = np.zeros([N,H,W,1]);
			
			for index in range(int(np.ceil(N/batch_size))):
				slot = np.arange(index*batch_size,(index+1)*batch_size);
				if ((index+1)*batch_size > N):
					break
				batch_I = vid[slot];
				[processed_I] = sess.run([normalized_I],feed_dict={I:batch_I});
				processed_vid[slot] = processed_I;
		print(processed_vid)
		np.save(output_fnm,processed_vid);		
	
