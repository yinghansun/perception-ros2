from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class PlaneLabel(Enum):
    horizontal = 'horizontal', 1, 'red'
    vertical = 'vertical', 2, 'blue'
    sloping = 'sloping', 3, 'green'
    others = 'others', 4, 'yellow'

    def __init__(self, name: str, label: int, color: str) -> None:
        self.__name = name
        self.__label = label
        self.__color = color

    @property
    def name(self):
        return self.__name

    @property
    def label(self):
        return self.__label

    @property
    def color(self):
        return self.__color


def create_horizontal_plane(
    x_upper: float,
    x_lower: float,
    y_upper: float,
    y_lower: float,
    z: float,
    scale: Optional[float] = 0.03,
    label: Optional[bool] = False,
    visualization: Optional[bool] = False
) -> np.ndarray:
    num_points_x = int((x_upper - x_lower) / scale)
    num_points_y = int((y_upper - y_lower) / scale)

    x_list = np.linspace(x_upper, x_lower, num_points_x)
    y_list = np.linspace(y_upper, y_lower, num_points_y)
    
    if label:
        point_list = np.zeros((num_points_x * num_points_y, 4))
    else:
        point_list = np.zeros((num_points_x * num_points_y, 3))
    idx = 0
    for x in x_list:
        for y in y_list:
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.horizontal.label
            idx += 1

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2], c=PlaneLabel.horizontal.color)
        plt.gca().set_box_aspect((num_points_x, num_points_y, (num_points_x + num_points_y) / 2))
        plt.show()

    return point_list


def create_vertical_plane_xfixed(
    y_upper: float,
    y_lower: float,
    z_upper: float,
    z_lower: float,
    x: float,
    scale: Optional[float] = 0.03,
    label: Optional[bool] = False,
    visualization: Optional[bool] = False
) -> np.ndarray:
    num_points_y = int((y_upper - y_lower) / scale)
    num_points_z = int((z_upper - z_lower) / scale)

    y_list = np.linspace(y_upper, y_lower, num_points_y)
    z_list = np.linspace(z_upper, z_lower, num_points_z)
    
    if label:
        point_list = np.zeros((num_points_y * num_points_z, 4))
    else:
        point_list = np.zeros((num_points_y * num_points_z, 3))
    idx = 0
    for y in y_list:
        for z in z_list:
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.vertical.label
            idx += 1

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2], c=PlaneLabel.vertical.color)
        plt.gca().set_box_aspect(((num_points_y + num_points_z) / 2, num_points_y, num_points_z))
        plt.show()

    return point_list


def create_vertical_plane_yfixed(
    x_upper: float,
    x_lower: float,
    z_upper: float,
    z_lower: float,
    y: float,
    scale: Optional[float] = 0.03,
    label: Optional[bool] = False,
    visualization: Optional[bool] = False
) -> np.ndarray:
    num_points_x = int((x_upper - x_lower) / scale)
    num_points_z = int((z_upper - z_lower) / scale)

    x_list = np.linspace(x_upper, x_lower, num_points_x)
    z_list = np.linspace(z_upper, z_lower, num_points_z)
    
    if label:
        point_list = np.zeros((num_points_x * num_points_z, 4))
    else:
        point_list = np.zeros((num_points_x * num_points_z, 3))
    idx = 0
    for x in x_list:
        for z in z_list:
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.vertical.label
            idx += 1

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2], c=PlaneLabel.vertical.color)
        plt.gca().set_box_aspect((num_points_x, (num_points_x + num_points_z) / 2, num_points_z))
        plt.show()

    return point_list


def create_box(
    length: float,
    width: float,
    height: float,
    center: Optional[np.ndarray] = np.zeros(3),
    label: Optional[bool] = False,
    visualization: Optional[bool] = False
) -> np.ndarray:
    z_top = center[2] + height / 2
    z_bottom = center[2] - height / 2
    y_right = center[1] + width / 2
    y_left = center[1] - width / 2
    x_front = center[0] + length / 2
    x_rear = center[0] - length / 2

    top_point_list = create_horizontal_plane(x_front, x_rear, y_right, y_left, z_top, label=label)
    bottom_point_list = create_horizontal_plane(x_front, x_rear, y_right, y_left, z_bottom, label=label)
    right_point_list = create_vertical_plane_yfixed(x_front, x_rear, z_top, z_bottom, y_right, label=label)
    left_point_list = create_vertical_plane_yfixed(x_front, x_rear, z_top, z_bottom, y_left, label=label)
    front_point_list = create_vertical_plane_xfixed(y_right, y_left, z_top, z_bottom, x_front, label=label)
    rear_point_list = create_vertical_plane_xfixed(y_right, y_left, z_top, z_bottom, x_rear, label=label)

    point_list_list = [top_point_list, bottom_point_list, right_point_list, left_point_list, front_point_list, rear_point_list]
    total_num_points = 0
    for list in point_list_list:
        total_num_points += list.shape[0]
    if label:
        point_list = np.zeros((total_num_points, 4))
    else:
        point_list = np.zeros((total_num_points, 3))
    pointer = 0
    for list in point_list_list:
        point_list[pointer:pointer+list.shape[0], :] = list
        pointer += list.shape[0]
    
    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2])
        plt.gca().set_box_aspect((1, 1, 1))
        plt.xlabel('x')
        plt.ylabel('y')      
        plt.show()

    return point_list


def create_stairs(
    num_steps: int, 
    length: float, 
    width: float, 
    height: float,
    label: Optional[bool] = False,
    visualization: Optional[bool] = False
) -> np.ndarray:
    '''
    Paras:
    - num_steps: number of steps of the stair
    - length: length per steps
    - width: width per steps
    - height: height per steps
    '''
    cur_height = 0
    cur_length = 0
    point_list_list = []
    total_num_points = 0

    for _ in range(num_steps):
        cur_vertical_pointlist = create_vertical_plane_xfixed(width/2, -width/2, cur_height+height, cur_height, cur_length, label=label)
        point_list_list.append(cur_vertical_pointlist)
        total_num_points += cur_vertical_pointlist.shape[0]
        cur_height += height

        cur_horizontal_pointlist = create_horizontal_plane(cur_length+length, cur_length, width/2, -width/2, cur_height, label=label)
        point_list_list.append(cur_horizontal_pointlist)
        total_num_points += cur_horizontal_pointlist.shape[0]
        cur_length += length

    if label:
        point_list = np.zeros((total_num_points, 4))
    else:
        point_list = np.zeros((total_num_points, 3))
    pointer = 0
    for list in point_list_list:
        point_list[pointer:pointer+list.shape[0], :] = list
        pointer += list.shape[0]

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2])
        plt.gca().set_box_aspect((1, 1, 1))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return point_list


def add_noise_pointlist(
    point_list: np.ndarray, 
    std: float, 
    visualization: Optional[bool] = False
) -> np.ndarray:
    num_data = point_list.shape[0]
    noise = np.random.normal(0, std, num_data*3)
    print(noise.shape)

    for i in range(num_data):
        point_list[i, 0] += noise[3*i]
        point_list[i, 1] += noise[3*i+1]
        point_list[i, 2] += noise[3*i+2]

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2])
        plt.gca().set_box_aspect((1, 1, 1))
        plt.xlabel('x')
        plt.ylabel('y')      
        plt.show()

    return point_list 

if __name__ == '__main__':
    # point_list = create_horizontal_plane(0.5, -0.5, -0.2, -0.4, 0.3, label=True, visualization=True)
    # point_list = create_vertical_plane_xfixed(0.5, -0.5, -0.2, -0.4, 0.3, label=True, visualization=True)
    # point_list = create_vertical_plane_yfixed(0.5, -0.5, -0.2, -0.4, 0.3, label=True, visualization=True)
    # point_list = create_box(1, 1, 0.5, label=True, visualization=True)
    point_list = create_stairs(4, 0.3, 1, 0.25, label=True, visualization=True)
    point_list = add_noise_pointlist(point_list, 0.01, True)