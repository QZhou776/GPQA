import numpy as np
import open3d as o3d
import os
import time
import math

def pcd_to_cube(points, pad_size=0, colors=None, cube_size=None):
    '''
    :param points: a set of 3D points (n*3 numpy array)
    :param pad_size: padding size
    :param colors: a set of rgb attribute (n*3 numpy array)
    :param cube_size: the size of voxelized cube ([x_size, y_size, z_size])
    :return: [point cube] or [point cube + color cube]
    '''
    if cube_size is None:
        x_size = np.max(points[:, 0]).astype("int") + 1
        y_size = np.max(points[:, 1]).astype("int") + 1
        z_size = np.max(points[:, 2]).astype("int") + 1
    else:
        x_size = cube_size[0]
        y_size = cube_size[1]
        z_size = cube_size[2]
    print(x_size, y_size, z_size)
    point_cube = np.zeros((x_size+2*pad_size, y_size+2*pad_size, z_size+2*pad_size), dtype=np.int8)
    pcd_coor = points.astype("int") + pad_size
    point_cube[pcd_coor[:, 0], pcd_coor[:, 1], pcd_coor[:, 2]] = 1
    if colors is None:
        return point_cube
    else:
        color_cube = np.zeros((x_size+2*pad_size, y_size+2*pad_size, z_size+2*pad_size, 3), dtype=np.float32)
        for p_id in range(points.shape[0]):
            color_cube[pcd_coor[p_id, 0], pcd_coor[p_id, 1], pcd_coor[p_id, 2], :] = colors[p_id]
        return point_cube, color_cube

def cube_to_pcd(point_cube, pad_size=0, color_cube=None):
    '''
    :param point_cube: a 3D 0-1 matrix (numpy array x_size*y_size*z_size)
    :param color_cube: a 4D rgb matrix (numpy array x_size*y_size*z_size*3)
    :return: pcd data
    '''
    pcd = o3d.geometry.PointCloud()
    points = np.argwhere(point_cube != 0)
    if color_cube is not None:
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)
        for p_id in range(points.shape[0]):
            colors[p_id] = color_cube[points[p_id, 0], points[p_id, 1], points[p_id, 2]]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.points = o3d.utility.Vector3dVector(points-pad_size)
    return pcd

def write_log(log_path, info_list):
    '''

    :param log_path:
    :param info_list:
    :return:
    '''
    with open(log_path, 'a') as f:
        f.write(time.asctime() + ': ')
        for s in info_list:
            f.write(str(s))
            f.write(' ')
        f.write('\n')

def cut_pcd(start_point, box_shape, points, colors):
    [x0, y0, z0] = [start_point[0], start_point[1], start_point[2]]
    [x_size, y_size, z_size] = [box_shape[0], box_shape[1], box_shape[2]]
    point_cube, color_cube = pcd_to_cube(points, pad_size=0, colors=colors)
    print("Input shape: ", point_cube.shape, color_cube.shape)
    cut_point_cube = point_cube[x0:x0 + x_size, y0:y0 + y_size, z0:z0 + z_size]
    cut_color_cube = color_cube[x0:x0 + x_size, y0:y0 + y_size, z0:z0 + z_size, :]
    print(cut_point_cube.shape)
    cut_pcd = cube_to_pcd(point_cube=cut_point_cube, pad_size=0, color_cube=cut_color_cube)
    return cut_pcd



def get_point_color_density(rad, points, colors=None):
    '''
    processing each occupied point, do not add points
    :param rad: search radius, search space: (2*rad+1)^3
    :param points: a set of 3D points (n*3 numpy array)
    :param colors: a set of rgb attribute (n*3 numpy array)
    :return: pcd with blured attribute
    '''
    quant_rad = math.ceil(rad)
    pad_size = math.ceil(rad)
    if colors is None:
        point_cube = pcd_to_cube(points, pad_size=pad_size)
    else:
        point_cube, color_cube = pcd_to_cube(points, pad_size=pad_size, colors=colors)
    point_num = points.shape[0]
    [x_size, y_size, z_size] = point_cube.shape
    print(x_size, y_size, z_size)
    pcd_coor = points.astype("int")

    point_density = np.zeros(point_cube.shape, dtype=np.int8)
    if colors is None:
        # processing geometry
        for p_id in range(point_num):
            [cur_x, cur_y, cur_z] = pcd_coor[p_id] + pad_size
            p_sum = 0
            for x_offset in range(-quant_rad, quant_rad + 1):
                for y_offset in range(-quant_rad, quant_rad + 1):
                    for z_offset in range(-quant_rad, quant_rad + 1):
                        p_sum += point_cube[cur_x + x_offset, cur_y + y_offset, cur_z + z_offset]
            point_density[cur_x, cur_y, cur_z] = p_sum
        # return point_density
    else:
        # processing color
        color_density = np.zeros(color_cube.shape, dtype=np.float32)
        for p_id in range(point_num):
            if p_id % 1000 == 0:
                print(p_id)
            [cur_x, cur_y, cur_z] = pcd_coor[p_id] + pad_size
            rgb_sum = np.array([0, 0, 0], dtype=np.float64)
            p_sum = 0
            for x_offset in range(-quant_rad, quant_rad + 1):
                for y_offset in range(-quant_rad, quant_rad + 1):
                    for z_offset in range(-quant_rad, quant_rad + 1):
                        p_sum += point_cube[cur_x + x_offset, cur_y + y_offset, cur_z + z_offset]
                        # print(color_cube[cur_x+x_offset, cur_y+y_offset, cur_z+z_offset, :].dtype, rgb_sum.dtype)
                        rgb_sum += color_cube[cur_x + x_offset, cur_y + y_offset, cur_z + z_offset, :]
            point_density[cur_x, cur_y, cur_z] = p_sum
            color_density[cur_x, cur_y, cur_z] = rgb_sum / (p_sum * 1.0)
        # return point_density color_density
        # out_pcd = cube_to_pcd(point_density, pad_size=pad_size, color_cube=color_density)
        del points, point_cube, color_cube
        return point_density, color_density

def get_point_color_density_v1(rad, blur_rad, points, colors=None):
    '''
    blurring each occupied point using density, adding points
    :param rad: search radius, search space: (2*rad+1)^3
    :param points: a set of 3D points (n*3 numpy array)
    :param colors: a set of rgb attribute (n*3 numpy array)
    :return: pcd with blurred attribute and blurred geometry
    '''
    quant_rad = math.ceil(rad)
    quant_blur_rad = math.ceil(blur_rad)
    pad_size = quant_rad + quant_blur_rad  # avoid overflow
    if colors is None:
        point_cube = pcd_to_cube(points, pad_size=pad_size)
    else:
        point_cube, color_cube = pcd_to_cube(points, pad_size=pad_size, colors=colors)
    point_num = points.shape[0]
    [x_size, y_size, z_size] = point_cube.shape
    print(x_size, y_size, z_size)
    pcd_coor = points.astype("int")

    point_density = np.zeros(point_cube.shape, dtype=np.int8)
    occu_cube = point_cube.copy()  # indicate whether the point needs to be visited
    if colors is None:
        # processing geometry
        for p_id in range(point_num):
            if p_id % 1000 == 0:
                print(p_id)
            [occ_x, occ_y, occ_z] = pcd_coor[p_id] + pad_size  # the point in the original input point cloud
            visit_point_list = []  # the point list to be visited in this round
            for x_offset in range(-quant_blur_rad, quant_blur_rad + 1):
                for y_offset in range(-quant_blur_rad, quant_blur_rad + 1):
                    for z_offset in range(-quant_blur_rad, quant_blur_rad + 1):
                        [cur_x, cur_y, cur_z] = [occ_x+x_offset, occ_y+y_offset, occ_z+z_offset]
                        if occu_cube[cur_x, cur_y, cur_z] == 0:
                            visit_point_list.append([cur_x, cur_y, cur_z])
                            occu_cube[cur_x, cur_y, cur_z] = 1  # mark visited

            for [cur_x, cur_y, cur_z] in visit_point_list:
                p_sum = 0
                for x_offset in range(-quant_rad, quant_rad + 1):
                    for y_offset in range(-quant_rad, quant_rad + 1):
                        for z_offset in range(-quant_rad, quant_rad + 1):
                            p_sum += point_cube[cur_x + x_offset, cur_y + y_offset, cur_z + z_offset]
                point_density[cur_x, cur_y, cur_z] = p_sum
        del points, point_cube, color_cube, occu_cube
        return point_density
    else:
        # processing color
        color_density = np.zeros(color_cube.shape, dtype=np.float32)
        for p_id in range(point_num):
            if p_id % 1000 == 0:
                print(p_id)

            [occ_x, occ_y, occ_z] = pcd_coor[p_id] + pad_size  # the point in the original input point cloud
            visit_point_list = []  # the point list to be visited in this round
            for x_offset in range(-quant_blur_rad, quant_blur_rad + 1):
                for y_offset in range(-quant_blur_rad, quant_blur_rad + 1):
                    for z_offset in range(-quant_blur_rad, quant_blur_rad + 1):
                        [cur_x, cur_y, cur_z] = [occ_x + x_offset, occ_y + y_offset, occ_z + z_offset]
                        if occu_cube[cur_x, cur_y, cur_z] == 0:
                            visit_point_list.append([cur_x, cur_y, cur_z])
                            occu_cube[cur_x, cur_y, cur_z] = 1  # mark visited

            for [cur_x, cur_y, cur_z] in visit_point_list:
                rgb_sum = np.array([0, 0, 0], dtype=np.float64)
                p_sum = 0
                for x_offset in range(-quant_rad, quant_rad + 1):
                    for y_offset in range(-quant_rad, quant_rad + 1):
                        for z_offset in range(-quant_rad, quant_rad + 1):
                            p_sum += point_cube[cur_x + x_offset, cur_y + y_offset, cur_z + z_offset]
                            # print(color_cube[cur_x+x_offset, cur_y+y_offset, cur_z+z_offset, :].dtype, rgb_sum.dtype)
                            rgb_sum += color_cube[cur_x + x_offset, cur_y + y_offset, cur_z + z_offset, :]
                point_density[cur_x, cur_y, cur_z] = p_sum
                color_density[cur_x, cur_y, cur_z] = rgb_sum / (p_sum * 1.0)
        # return point_density color_density
        # out_pcd = cube_to_pcd(point_density, pad_size=pad_size, color_cube=color_density)
        del points, point_cube, color_cube, occu_cube
        return point_density, color_density

