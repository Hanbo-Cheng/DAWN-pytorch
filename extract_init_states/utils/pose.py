# coding: utf-8

"""
Reference: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py

Calculating pose from the output 3DMM parameters, you can also try to use solvePnP to perform estimation
"""

__author__ = 'cleardusk'

import cv2
import numpy as np
from math import cos, sin, atan2, asin, sqrt

from .functions import calc_hypotenuse, plot_image


def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3] # shift
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0  #
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2) # r1r2，r3

    R = np.concatenate((r1, r2, r3), 0)  #  r 1-3R （）
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z

def angle2matrix(theta):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    R_x = np.array([[1,         0,                  0         ],

                    [0,         cos(theta[1]), -sin(theta[1]) ],

                    [0,         sin(theta[1]), cos(theta[1])  ]

                    ])

 

    R_y = np.array([[cos(theta[0]),    0,      sin(-theta[0])  ],

                    [0,                     1,      0         ],

                    [-sin(-theta[0]),   0,      cos(theta[0])  ]

                    ])

 

    R_z = np.array([[cos(theta[2]),    -sin(theta[2]),    0],

                    [sin(theta[2]),    cos(theta[2]),     0],

                    [0,                     0,            1]

                    ])

 

    R = np.dot(R_z, np.dot( R_y, R_x ))

 

    return R

def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw. 
        z: roll. 
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[1], angles[0], angles[2]
    
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  sin(x)],
                 [0, -sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, -sin(y)],
                 [      0, 1,      0],
                 [sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), sin(z), 0],
                 [-sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)

def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]

    return P, pose


def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    return point_3d


def plot_pose_box(img, P, ver, color=(40, 255, 0), line_width=2):
    """ Draw a 3D box as annotation of pose.
    Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        img: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (2, 68) or (3, 68)
    """
    llength = calc_hypotenuse(ver)
    point_3d = build_camera_box(llength)
    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(ver[:2, :27], 1)  # lmk 0-27 
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return img


def viz_pose(img, param_lst, ver_lst, show_flag=False, wfp=None):
    for param, ver in zip(param_lst, ver_lst):
        P, pose = calc_pose(param)
        img = plot_pose_box(img, P, ver)
        # print(P[:, :3])
        # print(f'yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')

    if wfp is not None:
        cv2.imwrite(wfp, img)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(img)

    return img

def pose_6(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)  # Convert the rotation matrix R to Euler angle form to obtain the pose.
    # print(t3d)
    R1 = angle2matrix(pose)
    # print(R)
    # print(R1)
    pose = [p * 180 / np.pi for p in pose]
    
    return s, pose, t3d, P  # s(scale)、R(roate)、t3d(shift)


def smooth_pose(img, param_lst, ver_lst, pose_new, show_flag=False, wfp=None, wnp = None):
    for param, ver in zip(param_lst, ver_lst):
        t3d = np.array([pose_new[4],pose_new[5],pose_new[6]])
        
        theta = np.array([pose_new[0],pose_new[1],pose_new[2]])
        theta = [p * np.pi / 180 for p in theta]
        R = angle2matrix(theta)
        P = np.concatenate((R, t3d.reshape(3, -1)), axis=1) 
        img = plot_pose_box(img, P, ver)
    #    print(P,P.shape,t3d)
        # print(P,pose_new)
        # print(f'yaw: {theta[0]:.1f}, pitch: {theta[1]:.1f}, roll: {theta[2]:.1f}')
        all_pose = [0]
        all_pose = np.array(all_pose)

    if wfp is not None:
        cv2.imwrite(wfp, img)
        print(f'Save visualization result to {wfp}')
        
    if wnp is not None:
        np.save(wnp, all_pose)
        print(f'Save visualization result to {wfp}')
        
    if show_flag:
        plot_image(img)

    return img

    
    
    

def get_pose(img, param_lst, ver_lst, show_flag=False, wfp=None, wnp = None):
    for param, ver in zip(param_lst, ver_lst):  # only one loop
        s, pose, t3d, P = pose_6(param)
        img_1 = plot_pose_box(img.copy(), P, ver)
    #    print(P,P.shape,t3d)
        # print(f'yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')
        all_pose = [pose[0],pose[1],pose[2],s,t3d[0],t3d[1],t3d[2]]
        all_pose = np.array(all_pose)

    # if wfp is not None:
    #     cv2.imwrite(wfp, img_1)
    #     print(f'Save visualization result to {wfp}')
        
    # if wnp is not None:
    #     np.save(wnp, all_pose)
    #     print(f'Save visualization result to {wfp}')
        
    if show_flag:
        plot_image(img)

    return all_pose
