from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def create_horizontal_plane(
    x_upper: float,
    x_lower: float,
    y_upper: float,
    y_lower: float,
    z: float,
    scale: Optional[float] = 0.03,
    visualization: Optional[bool] = False
) -> np.ndarray:
    num_points_x = int((x_upper - x_lower) / scale)
    num_points_y = int((y_upper - y_lower) / scale)

    x_list = np.linspace(x_upper, x_lower, num_points_x)
    y_list = np.linspace(y_upper, y_lower, num_points_y)
    
    point_list = np.zeros((num_points_x * num_points_y, 3))
    idx = 0
    for x in x_list:
        for y in y_list:
            point_list[idx, :] = np.array([x, y, z])
            idx += 1

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2])
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
    visualization: Optional[bool] = False
) -> np.ndarray:
    num_points_y = int((y_upper - y_lower) / scale)
    num_points_z = int((z_upper - z_lower) / scale)

    y_list = np.linspace(y_upper, y_lower, num_points_y)
    z_list = np.linspace(z_upper, z_lower, num_points_z)
    
    point_list = np.zeros((num_points_y * num_points_z, 3))
    idx = 0
    for y in y_list:
        for z in z_list:
            point_list[idx, :] = np.array([x, y, z])
            idx += 1

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2])
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
    visualization: Optional[bool] = False
) -> np.ndarray:
    num_points_x = int((x_upper - x_lower) / scale)
    num_points_z = int((z_upper - z_lower) / scale)

    x_list = np.linspace(x_upper, x_lower, num_points_x)
    z_list = np.linspace(z_upper, z_lower, num_points_z)
    
    point_list = np.zeros((num_points_x * num_points_z, 3))
    idx = 0
    for x in x_list:
        for z in z_list:
            point_list[idx, :] = np.array([x, y, z])
            idx += 1

    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2])
        plt.gca().set_box_aspect((num_points_x, (num_points_x + num_points_z) / 2, num_points_z))
        plt.show()

    return point_list


def create_box(
    length: float,
    width: float,
    height: float,
    center: Optional[np.ndarray] = np.zeros(3),
    visualization: Optional[bool] = False
):
    z_top = center[2] + height / 2
    z_bottom = center[2] - height / 2
    y_right = center[1] + width / 2
    y_left = center[1] - width / 2
    x_front = center[0] + length / 2
    x_rear = center[0] - length / 2

    top_point_list = create_horizontal_plane(x_front, x_rear, y_right, y_left, z_top)
    bottom_point_list = create_horizontal_plane(x_front, x_rear, y_right, y_left, z_bottom)
    right_point_list = create_vertical_plane_yfixed(x_front, x_rear, z_top, z_bottom, y_right)
    left_point_list = create_vertical_plane_yfixed(x_front, x_rear, z_top, z_bottom, y_left)
    front_point_list = create_vertical_plane_xfixed(y_right, y_left, z_top, z_bottom, x_front)
    rear_point_list = create_vertical_plane_xfixed(y_right, y_left, z_top, z_bottom, x_rear)

    point_list_list = [top_point_list, bottom_point_list, right_point_list, left_point_list, front_point_list, rear_point_list]
    total_num_points = 0
    for list in point_list_list:
        total_num_points += list.shape[0]
    point_list = np.zeros((total_num_points, 3))
    pointer = 0
    for list in point_list_list:
        point_list[pointer:pointer+list.shape[0], :] = list
        pointer += list.shape[0]
    
    if visualization:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(point_list[:, 0], point_list[:, 1], point_list[:, 2])
        plt.gca().set_box_aspect((1, 1, 1))
        plt.show()

    return point_list


def create_stairs(num_steps: int, width: float, height: float):
    '''
    Paras:
    - num_steps: number of steps of the stair
    - width: width per steps
    - height: height per steps
    '''

if __name__ == '__main__':
    # create_horizontal_plane(0.5, -0.5, -0.2, -0.4, 0.3, visualization=True)
    # create_vertical_plane_xfixed(0.5, -0.5, -0.2, -0.4, 0.3, visualization=True)
    # create_vertical_plane_yfixed(0.5, -0.5, -0.2, -0.4, 0.3, visualization=True)
    point_list = create_box(1, 1, 0.5, visualization=True)
