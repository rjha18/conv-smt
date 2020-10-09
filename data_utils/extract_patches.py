import scipy.io as sio
import numpy as np
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt

N = 10000
K = 12

images = np.array(sio.loadmat("../Data/IMAGES.mat")['IMAGES'])

patches = []

for i in range(10):
    ptch = image.extract_patches_2d(images[:, :, i], (K, K), max_patches=int(N / 10))
    patches.append(ptch)

patches = np.array(patches)
patches = patches.reshape([-1, K, K])
np.random.shuffle(patches)
print(images.shape)
np.save("../top_k/images.npy", patches)

plt.imshow(patches[0, :, :], cmap='gray')
plt.show()
plt.savefig("abc.png")