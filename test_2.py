import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import try_all_threshold,threshold_otsu,threshold_mean

img = data.page()

fig, ax = threshold_mean(img)
plt.show()