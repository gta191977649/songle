# importing the numpy module
from PIL import Image
import numpy as np
import utils as helper
image = Image.open("1.png").convert('L')

data = np.array(image)
filter = helper.b_spline()

filt_len = filter.shape[0]
L = int((filt_len-1)/2)
new_array = np.zeros(shape=(2*L+data.shape[0],data.shape[1]))
for i in range(0,data.shape[0]):
    for j in range(0,data.shape[1]):
        if(i < L):
            new_array[i,j] = data[L-1+i,j]
        if(i > L+data.shape[0]):
            new_array[i,j] = data[L+data.shape[0]-1-i,j]
        if(i >= L and i <= L+data.shape[0]):
            new_array[i,j] = data[i,j]

#data_processed = helper.vertFilter(data,filter)

data_processed = helper.horiFilter(helper.vertFilter(data,filter),filter)
#
# print(filter)
pil_image=Image.fromarray(data_processed)
pil_image.show()