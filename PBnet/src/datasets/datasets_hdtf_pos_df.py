from os import name
from src.datasets.datasets_hdtf_pos import HDTF
import sys
sys.path.append('your_path')

import os
import random
import torch

import numpy as np
import torch.utils.data as data
import imageio.v2 as imageio

import cv2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d

# from ..utils.tensors import collate

def resize(im, desired_size, interpolation):
    old_size = im.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple(int(x*ratio) for x in old_size)

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=interpolation)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im

class HDTF(data.Dataset):
    def __init__(self, data_dir, max_num_frames=80, min_num_frames=40, mode='train'):

        super(HDTF, self).__init__()
        
        self.data_dir = data_dir
        self.max_num_frames = max_num_frames
        self.min_num_frames = min_num_frames
        self.mode = mode
        # self.hubert_dir = '/train20/intern/permanent/lmlin2/data/crema_wav_hubert'
        # self.pose_dir = '/train20/intern/permanent/hbcheng2/data/crema/pose'

        self.pose_dir = '/train20/intern/permanent/hbcheng2/data/HDTF/pose_bar'
        self.hubert_dir = '/train20/intern/permanent/lmlin2/data/hdtf_wav_hubert'
        
        vid_list = []
        # # crema
        # if mode == 'train':
        #     for id_name in os.listdir(data_dir):
        #         if id_name in ['s15','s20','s21','s30','s33','s52','s62','s81','s82','s89']: #['s64','s76','s88','s90','s91']
        #             continue
        #         vid_list.extend([os.path.join(id_name, sent) for sent in os.listdir(f'{data_dir}/{id_name}') ])

        # # crema
        # if mode == 'test':
        #     for id_name in ['s15','s20','s21','s30','s33','s52','s62','s81','s82','s89']:
        #         vid_list.extend([os.path.join(id_name, sent) for sent in os.listdir(f'{data_dir}/{id_name}') ])



        # hdtf  
        vid_id_name_list = ['RD_Radio14_000','RD_Radio30_000','RD_Radio47_000','RD_Radio56_000','WDA_AmyKlobuchar1_001',\
                            'WDA_BarbaraLee0_000','WDA_BobCasey0_000','WDA_CatherineCortezMasto_000','WDA_DebbieDingell1_000','WDA_DonaldMcEachin_000',\
                            'WDA_EricSwalwell_000','WDA_HenryWaxman_000','WDA_JanSchakowsky1_000','WDA_JoeDonnelly_000','WDA_JohnSarbanes1_000',\
                            'WDA_JoeNeguse_001','WDA_KatieHill_000','WDA_LucyMcBath_000','WDA_MazieHirono0_000','WDA_NancyPelosi1_000',\
                            'WDA_PattyMurray0_000','WDA_RaulRuiz_000','WDA_SeanPatrickMaloney_000','WDA_TammyBaldwin0_000','WDA_TerriSewell0_000',\
                            'WDA_TomCarper_000','WDA_WhipJimClyburn_000','WRA_AdamKinzinger0_000','WRA_AnnWagner_000','WRA_BobCorker_000',\
                            'WRA_CandiceMiller0_000','WRA_CathyMcMorrisRodgers2_000','WRA_CoryGardner1_000','WRA_DebFischer1_000','WRA_DianeBlack1_000',\
                            'WRA_ErikPaulsen_000','WRA_GeorgeLeMieux_000','WRA_JebHensarling0_001','WRA_JoeHeck1_000','WRA_JohnKasich1_001',\
                            'WRA_MarcoRubio_000']
        bad_id_name_list = ['WDA_DanKildee_000', 'WDA_PatrickLeahy1_000', 'WRA_KristiNoem2_000', 'RD_Radio39_000']
        if mode == 'train':
            for id_name in os.listdir(data_dir):
                if id_name in vid_id_name_list or id_name in bad_id_name_list:
                    continue
                vid_list.append(id_name)
            self.videos = vid_list
        if mode == 'test':
            self.videos = vid_id_name_list


    # def __len__(self):
    #     return len(self.videos)

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        return min(len(self.videos), num_seq_max)
        
    def __getitem__(self, idx):
        
        video_name = self.videos[idx]
        path = os.path.join(self.data_dir, video_name)
        # path_pose = os.path.join(self.pose_dir, video_name)

        frame_path_list = os.listdir(path)
        frame_path_list.sort()
        total_num_frames = len(frame_path_list)

        # pose_path_list = os.listdir(path_pose)
        # pose_path_list.sort()

        hubert_path = os.path.join(self.hubert_dir, video_name+'.npy')
        hubert_feature = np.load(hubert_path)
        Nframes_hubert = hubert_feature.shape[0]
        interp_func = interp1d(np.arange(Nframes_hubert), hubert_feature, kind='linear', axis=0)
        hubert_feature = interp_func(np.linspace(0, Nframes_hubert - 1, total_num_frames)).astype(np.float32)
        
        pose_path = os.path.join(self.pose_dir, video_name+'.npy')
        pose_seq = np.load(pose_path).astype(np.float32)

        cur_num_frames = np.random.randint(self.min_num_frames, self.max_num_frames+1)
        if total_num_frames <= cur_num_frames:
            sample_frames = total_num_frames
            start = 0
        else:
            sample_frames = cur_num_frames
            start = np.random.randint(total_num_frames-cur_num_frames)
        sample_idx_list = np.linspace(start=start, stop=sample_frames+start-1, num=sample_frames, dtype=int)
        # sample_frame_path_list = [frame_path_list[x] for x in sample_idx_list]
        # sample_pose_path_list = [pose_path_list[x] for x in sample_idx_list]

        sample_hubert_feature_list = [hubert_feature[x,:] for x in sample_idx_list]  # nf,1024
        sample_hubert_feature_tensor = [torch.from_numpy(arr) for arr in sample_hubert_feature_list]
        sample_hubert_feature_tensor = torch.stack(sample_hubert_feature_tensor)
        # sample_hubert_feature_list = np.stack(sample_hubert_feature_list).reshape(-1)  # (nf*1024)

        # load pose
        try:
            # sample_pose_list = [np.load(os.path.join(path_pose, x))[0][:-1].astype(np.float32) for x in sample_pose_path_list]
            sample_pose_list = [pose_seq[x,:] for x in sample_idx_list]
            sample_pos_feature_tensor = [torch.from_numpy(arr) for arr in sample_pose_list]
            sample_pos_feature_tensor = torch.stack(sample_pos_feature_tensor) # nf, 6
        except Exception:
            # print(os.path.join(path_pose, x))
            print("load fail !! ")
            print(pose_path)
            print(sample_idx_list)
            sample_pose_list = [pose_seq[x,:] for x in sample_idx_list]
            sample_pos_feature_tensor = [torch.from_numpy(arr) for arr in sample_pose_list]
            sample_pos_feature_tensor = torch.stack(sample_pos_feature_tensor) # nf, 6
        
        # added to change the video_name of crema
        video_name = video_name.replace('/','_')
        # sample_class_tensor = torch.tensor(0)

        return sample_hubert_feature_tensor, sample_pos_feature_tensor, video_name

    def update_parameters(self, parameters):
        _, self.pos_dim = self[0][1].shape
        _, self.audio_dim = self[0][0].shape
        parameters["audio_dim"] = self.audio_dim
        parameters["pos_dim"] = self.pos_dim
        # parameters["njoints"] = self.njoints


if __name__ == "__main__":
    data_dir = "/yrfs2/cv2/pcxia/audiovisual/hdtf/images_25hz"
    # data_dir='/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/images'
    dataset = HDTF(data_dir=data_dir,mode='test')
    for i in range(100):
        dataset.__getitem__(i)
        print('------')    

    test_dataset = data.DataLoader(dataset=dataset,
                                    batch_size=10,
                                    num_workers=8,
                                    shuffle=False)
    for i, batch in enumerate(test_dataset):
        print(i)
