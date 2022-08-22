import os
import numpy as np
from point_could_utils import *

cur_path = os.path.dirname(os.path.abspath(__file__))
data_root_path = cur_path + '/../data/'

stair1 = create_stairs(num_steps=4, length=0.3, width=1, height=0.25, label=True)
np.save(data_root_path + 'stair1.npy', stair1)

box1 = create_box(length=1, width=1, height=0.5, label=True)
np.save(data_root_path + 'box1.npy', box1)