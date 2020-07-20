import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt;
from PIL import Image;

frames = np.load('./Data/processed/e-processed.npy')
print (frames.shape)
plt.imshow(frames[-100,:,:,0],cmap='gray')
plt.show()