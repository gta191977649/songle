import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.filters import try_all_threshold,threshold_otsu,threshold_mean,threshold_multiotsu
from PIL import Image

img = data.camera()
#img = np.digitize(img, threshold_multiotsu(img))

thresh = threshold_otsu(img.ravel())

binary = img > thresh


pil_image=Image.fromarray(binary)
pil_image.show()