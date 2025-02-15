# coding: utf-8
# based on 3DDFA
__author__ = 'cleardusk'

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(current_dir)
import argparse
import cv2
import yaml

import time
from yaml import safe_dump
from FaceBoxes import FaceBoxes
import numpy as np
from tqdm import tqdm
import copy
import time
from utils.pose import viz_pose, get_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix, calculate_eye, calculate_bbox
from utils.tddfa_util import str2bool
import concurrent.futures
from multiprocessing import Pool

def main(args,img, save_path, pose_path):
 #   begin = time.time()
    
        # face_boxes.eval()

    # Given a still image path and load to BGR channel
  #  img = cv2.imread(img_path) #args.img_fp

    # Detect faces, get 3DMM params and roi boxes

    # start_time = time.time()
    boxes = face_boxes(img)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f'box time: {execution_time}')
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
      #  sys.exit(-1)
        return None
    # print(f'Detect {n} faces')

    # start_time = time.time()
    param_lst, roi_box_lst = tddfa(img, boxes)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f'tddfa time: {execution_time}')
    #detection time
  #  detect_time = time.time()-begin
 #   print('detection time: '+str(detect_time), file=open('/mnt/lustre/jixinya/Home/3DDFA_V2/pose.txt', 'a'))
    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
  #  old_suffix = get_suffix(img_path)
    old_suffix = 'png'
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    # start_time = time.time()
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f'tddfa.recon_vers time: {execution_time}')


    # start_time = time.time()
    all_pose = get_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=save_path, wnp = pose_path)
    end_time = time.time()
    

    return all_pose, ver_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default=f'{current_dir}/configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='/disk2/pfhu/DAWN-pytorch/images/image/anime_female2.jpeg')
    parser.add_argument('-m', '--mode', type=str, default='gpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='pose',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='False', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=True)
    parser.add_argument('-p', '--part',  type=int, default=1)
    parser.add_argument('-a', '--all', type=int, default=1)

    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-t', '--output', type=str)

    args = parser.parse_args()

    part = args.part
    all_part = args.all


    
    filepath = args.input
    save_path = args.output

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    start_point = 30 #int((part - 1) *duration)
    
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
        os.environ['OMP_NUM_THREADS'] = '8'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        # tddfa.eval()
        face_boxes = FaceBoxes()


    # save_path_pose = os.path.join(save_path, 'tmp.npy')
    image= cv2.imread(filepath)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    pose, lmk = main(args,image, save_path = None, pose_path =  None)

    lmk = lmk[0]
    eye_bbox_result = np.zeros(8)
    bbox = calculate_bbox(image, lmk)
    left_ratio, right_ratio = calculate_eye(lmk)
    eye_bbox_result[0] = left_ratio.sum()
    eye_bbox_result[1] = right_ratio.sum()
    eye_bbox_result[2:] = np.array(bbox)

    pose = pose.reshape(1,7)
    eye_bbox_result = eye_bbox_result.reshape(1, -1)
    eye_bbox_path = os.path.join(save_path, 'init_eye_bbox.npy')
    pose_path = os.path.join(save_path, 'init_pose.npy')

    np.save(eye_bbox_path, eye_bbox_result)
    np.save(pose_path, pose)


            
   