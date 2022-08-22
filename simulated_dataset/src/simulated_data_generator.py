import os
from venv import create
import numpy as np
from point_could_utils import *

cur_path = os.path.dirname(os.path.abspath(__file__))
data_root_path = cur_path + '/../data/'

stair1 = create_stairs(num_steps=4, length=0.3, width=1, height=0.25, label=True)
np.save(data_root_path + 'stair1.npy', stair1)


for i in range(10):
    name = 'box' + str(i) + '.npy'
    length = np.random.rand()
    width = np.random.rand()
    height = np.random.rand()

    if np.random.rand() > 0.5:
        length += 1
    else:
        length += 1.5

    if np.random.rand() > 0.5:
        width += 0.365
    else:
        width += 0.983

    if np.random.rand() > 0.5:
        height += 0.454
    else:
        height += 1.344

    box_i = create_box(length=length, width=width, height=height, label=True)
    angle = np.random.rand() * 3.1415926
    box_i = rotate_pointlist(box_i, angle)
    box_i = add_noise_pointlist(box_i, 0.005)
    # visualize_pointlist(box_i)
    np.save(data_root_path + name, box_i)

# print(np.random.rand())