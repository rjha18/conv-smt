import numpy as np
import matplotlib.pyplot as plt

frames = np.load('./Data/bear-processed.npy')
print(frames.shape)
plt.imshow(frames[150, :, :, 0], cmap='gray')
plt.show()
plt.savefig("./Data/preview/preview")
