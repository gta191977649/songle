# importing the numpy module
from PIL import Image
import numpy as np
import utils as helper
image = Image.open("1.png").convert('L')

data = np.array(image)
filter = helper.b_spline()

data_processed = helper.horiFilter(helper.vertFilter(data,filter),filter)

print(filter)
pil_image=Image.fromarray(data_processed)
pil_image.show()