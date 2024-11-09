from os import name
import sys
# sys.path.append('your_path')

import os
import random
import torch

import numpy as np
import torch.utils.data as data
import torch.nn.functional as Ft
import imageio.v2 as imageio

import cv2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
# import decord
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import time
import pickle as pkl

# decord.bridge.set_bridge('torch')


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
    def __init__(self, data_dir, max_num_frames=80, mode='train'):

        super(HDTF, self).__init__()
        self.data_dir = data_dir
        self.max_num_frames = max_num_frames
        self.mode = mode
            
        self.hubert_dir = '/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wav_hubert_interpolate'  #hdtf hubert 
        # self.hubert_dir =  '/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wavlm_interpolate_chunk'   
        self.pose_dir = '/train20/intern/permanent/hbcheng2/data/HDTF/pose_bar'
        self.eye_blink_dir = '/train20/intern/permanent/hbcheng2/data/HDTF/eye_blink_bbox_from_xpc_bar'

        # self.max_vals = torch.tensor([20, 10, 10,  7e-4,
        #     7e+1,  9e+1]).to(torch.float32)
        # self.min_vals = torch.tensor([-20, -10, -10,  4e-4,
        #     5e+1,  6e+1]).to(torch.float32)

        self.max_vals = torch.tensor([90, 90, 90,  1,
            720,  1080]).to(torch.float32)
        self.min_vals = torch.tensor([-90, -90, -90,  0,
            0,  0]).to(torch.float32)

        vid_list = []

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

        bad_id_name = ['WDA_DanKildee_000', 'WDA_PatrickLeahy1_000', 'WRA_KristiNoem2_000']

        

        # vid_id_name_list = [item + '.mp4' for item in vid_id_name_list]
        # bad_id_name = [item + '.mp4' for item in bad_id_name]

        with open('/train20/intern/permanent/hbcheng2/data/HDTF/length_dict.pkl', 'rb') as f:
            self.len_dict = pkl.load(f)

        if mode == 'train':
            for id_name in os.listdir(data_dir):
                # id_name = id_name[:-4]
                if id_name in vid_id_name_list or id_name in bad_id_name:
                    continue
                vid_list.append(id_name)
            self.videos = vid_list
        if mode == 'test':
            self.videos = vid_id_name_list

        self.cache_audio = {}
        self.cache_eye = {}
        self.cache_pose = {}
        
        for video in self.videos:
            hubert_path = os.path.join(self.hubert_dir, video) + '.npy'
            pose_path = os.path.join(self.pose_dir, video) + '.npy'
            eye_blink_path = os.path.join(self.eye_blink_dir, video) + '.npy'

            hubert_fea = np.load(hubert_path)
            pose_fea = np.load(pose_path)
            blink_fea = np.load(eye_blink_path)

            self.cache_audio[video] = hubert_fea
            self.cache_pose[video] = pose_fea
            self.cache_eye[video] = blink_fea

    def check_head(self, frame_list, video_name, start, end):
        start_path = self.get_pose_path(frame_list, video_name, start)
        end_path = self.get_pose_path(frame_list, video_name, end)

        if os.path.exists(start_path) and os.path.exists(end_path):
            return True
        else:
            return False


    def get_block_data_for_two(self, path, start, end):
        # TODO： id function
        '''
        input: 
            start: start id
            end:  end id
        output:
            the data from block
        '''

        block_st = start//25
        block_ed = end//25

        st_pos = block_st % 25
        ed_pos = block_ed % 25

        block_st_name = 'chunk_%04d.npy' % (block_st)
        block_ed_name = 'chunk_%04d.npy' % (block_ed)

        if block_st != block_ed:
            block_st_path = os.path.join(path, block_st_name)
            block_ed_path = os.path.join(path, block_ed_name)
            block_st = np.load(block_st_path)
            block_ed = np.load(block_ed_path)

            return np.concatenate((block_st[st_pos:], block_ed[:ed_pos]))
        else:
            block_st_path = os.path.join(path, block_st_name)
            block_st = np.load(block_st_path)
            return block_st[st_pos, ed_pos]

    def get_block_data(self, path, start, end):
        # TODO： id function
        '''
        input: 
            start: start id
            end:  end id
        output:
            the data from block
        '''

        block_st = start//25
        block_ed = end//25

        st_pos = start % 25
        ed_pos = end % 25

        block_list = [os.path.join(path,'chunk_%04d.npy' % (i)) for i in range(block_st, block_ed+1)]

        if block_st != block_ed:
            arr_list = []
            block_st = np.load(block_list[0])
            arr_list.append(block_st[st_pos:])
            for path in block_list[1:-1]:
                arr_list.append(np.load(path))

            block_ed = np.load(block_list[-1])
            arr_list.append(block_ed[:ed_pos])

            return np.concatenate(arr_list)
        else:
            block_st_path = os.path.join(path, block_list[0])
            block_st = np.load(block_st_path)
            return block_st[st_pos: ed_pos]
            

    def check_len(self, name):
        return self.len_dict[name]


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        # path = os.path.join(self.data_dir, video_name)
        # hubert_path = os.path.join(self.hubert_dir, video_name)
        # pose_path = os.path.join(self.pose_dir, video_name)
        # eye_blink_path = os.path.join(self.eye_blink_dir, video_name)

        total_num_frames = self.check_len(video_name)
        

        if total_num_frames <= self.max_num_frames:
            sample_frames = total_num_frames
            start = 0
        else:
            sample_frames = self.max_num_frames
            start = np.random.randint(total_num_frames-self.max_num_frames)
        start=start
        stop=sample_frames+start

        
        start_time = time.time()
        # sample_hubert_feature_npy = self.get_block_data(path = hubert_path, start = start, end = stop).astype(np.float32)
        # sample_pose_list_npy = self.get_block_data(path = pose_path, start = start, end = stop).astype(np.float32)

        sample_hubert_feature_npy = self.cache_audio[video_name][start:stop].astype(np.float32)
        sample_pose_list_npy = self.cache_pose[video_name][start:stop].astype(np.float32)
        sample_eye_blink_list_npy = self.cache_eye[video_name][start:stop].astype(np.float32)

        # end_time = time.time()
        # print("dataset_audiopose_cost: ", - start_time + end_time)
        # start_time = time.time()

        # sample_eye_blink_list_npy = self.get_block_data(path = eye_blink_path, start = start, end = stop).astype(np.float32)

        # end_time = time.time()
        # print("dataset_eye_cost: ", - start_time + end_time)
        # start_time = time.time()

        sample_hubert_feature_tensor = torch.tensor(sample_hubert_feature_npy)
        sample_pos_feature_tensor = torch.tensor(sample_pose_list_npy)[:,:-1]
        sample_pos_feature_tensor = (sample_pos_feature_tensor - self.min_vals)/ (self.max_vals - self.min_vals) 

        # end_time = time.time()
        # print("dataset_audiopose_cost2: ", - start_time + end_time)
        # start_time = time.time()

        sample_eye_feature_tensor = torch.tensor(sample_eye_blink_list_npy)[:,:2]

        # end_time = time.time()
        # print("dataset_eye_cost2: ", - start_time + end_time)
        # start_time = time.time()

        sample_pos_eye_cat_tensor  = torch.cat((sample_pos_feature_tensor,sample_eye_feature_tensor),dim=1)

        # end_time = time.time()
        # print("dataset_eye_cost3: ", - start_time + end_time)
        # start_time = time.time()

        
        video_name = video_name.replace('/','_')

        # sample_pose_list_npy = sample_pose_list_npy.transpose(1,0)  # for compatibility

        return sample_hubert_feature_tensor, sample_pos_feature_tensor, sample_eye_feature_tensor, video_name, start, sample_pos_eye_cat_tensor 

    def update_parameters(self, parameters):
        _, self.pos_dim = self[0][1].shape
        _, self.eye_dim = self[0][2].shape
        _, self.audio_dim = self[0][0].shape
        parameters["audio_dim"] = self.audio_dim
        parameters["pos_dim"] = self.pos_dim
        parameters["eye_dim"] = self.eye_dim
        # parameters["njoints"] = self.njoints




if __name__ == "__main__":
    # hdtf
    data_dir = "/yrfs2/cv2/pcxia/audiovisual/hdtf/images_25hz"
    # pose_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/pose"
    # crema
    # data_dir='/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/images'
    dataset = HDTF(data_dir=data_dir, mode='train')
    for i in range(10):
        dataset.__getitem__(i)
        print('------')    

    test_dataset = data.DataLoader(dataset=dataset,
                                    batch_size=10,
                                    num_workers=8,
                                    shuffle=False)
    for i, batch in enumerate(test_dataset):
        print(i)
