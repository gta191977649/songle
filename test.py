from PIL import Image
import numpy as np
import utils as helper
image = Image.open("img.jpg")

data = np.array(image)

filter = helper.b_spline()
data_processed = data - helper.horiFilter(helper.vertFilter(data,filter),filter)

print(data_processed)
pil_image=Image.fromarray(data_processed)
pil_image.show()