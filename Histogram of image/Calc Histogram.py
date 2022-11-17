# Import the necessary libraries
from PIL import Image as im
from numpy import asarray
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# load the image and convert into
# numpy array
img = im.open('hands.png')
imgGray = img.convert('L')

numpydata = np.array(imgGray)
plt.imshow(numpydata, cmap='gray')
plt.show()
numpydata = numpydata.reshape(-1)

info = dict(Counter(numpydata))

plt.bar(list(info.keys()), info.values(), color='g')
plt.show()
plt.savefig('result/part_two/hist.png')



