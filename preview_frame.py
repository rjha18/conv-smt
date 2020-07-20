import numpy as np
import matplotlib.pyplot as plt

frames = np.load('./Data/processed/e-processed.npy')
print(frames.shape)
plt.imshow(frames[-100, :, :, 0], cmap='gray')
plt.show()
