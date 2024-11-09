# dataset for HDTF, stage 2
from os import name
import sys
sys.path.append('your_path')

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
import decord
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import time
import pickle as pkl

decord.bridge.set_bridge('torch')


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
    def __init__(self, data_dir, pose_dir, eye_blink_dir, max_num_frames=80, image_size=128, audio_dir=None, ref_id = None, mode='train',
                 mean=(128, 128, 128), color_jitter=True):

        super(HDTF, self).__init__()
        self.mean = torch.tensor(mean)[None,:,None,None]
        self.data_dir = data_dir
        self.pose_dir = pose_dir
        self.eye_blink_dir = eye_blink_dir
        self.is_jitter = color_jitter
        self.max_num_frames = max_num_frames
        self.image_size = image_size
        self.mode = mode

        vid_list = []
        # # crema
        # self.hubert_dir = '/train20/intern/permanent/lmlin2/data/crema_wav_hubert'
        # if mode == 'train':
        #     for id_name in os.listdir(data_dir):
        #         if id_name in ['s64','s76','s88','s90','s91']:
        #             continue
        #         vid_list.extend([os.path.join(id_name, sent) for sent in os.listdir(f'{data_dir}/{id_name}') ])
        # if mode == 'test':
        #     for id_name in ['s64','s76','s88','s90','s91']:
        #         vid_list.extend([os.path.join(id_name, sent) for sent in os.listdir(f'{data_dir}/{id_name}') ])
        # self.videos = vid_list

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
        # hdtf  
        if audio_dir == None:
            self.hubert_dir = '/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wav_hubert_interpolate_chunk'  
        else:
            self.hubert_dir = audio_dir

        self.ref_id = ref_id
        self.mouth_dir = '/train20/intern/permanent/hbcheng2/data/HDTF/mouth_ratio_bar'
        self.lmk_dir = '/train20/intern/permanent/hbcheng2/data/HDTF/lmk_25hz_chunk'
        with open('/train20/intern/permanent/hbcheng2/data/HDTF/length_dict.pkl', 'rb') as f:
            self.len_dict = pkl.load(f)
        # vid_id_name_list = ['RD_Radio47_000','WDA_CatherineCortezMasto_000','WDA_JoeNeguse_001','WDA_MichelleLujanGrisham_000','WRA_ErikPaulsen_002', \
        #                     'WDA_ZoeLofgren_000','WRA_JebHensarling2_003','WRA_MichaelSteele_000', 'WRA_ToddYoung_000', 'WRA_VickyHartzler_000']
        if mode == 'train':
            for id_name in os.listdir(data_dir):
                # id_name = id_name[:-4]
                if id_name in vid_id_name_list or id_name in bad_id_name:
                    continue
                vid_list.append(id_name)
            self.videos = vid_list
        if mode == 'test':
            self.videos = vid_id_name_list

    def check_head(self, frame_list, video_name, start, end):
        '''
        Check if the desired pose address exists.
        '''
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
        path = os.path.join(self.data_dir, video_name)
        hubert_path = os.path.join(self.hubert_dir, video_name)
        lmk_path = os.path.join(self.lmk_dir, video_name)
        pose_path = os.path.join(self.pose_dir, video_name)
        eye_blink_path = os.path.join(self.eye_blink_dir, video_name)

        total_num_frames = self.check_len(video_name) 
        

        if total_num_frames <= self.max_num_frames:
            sample_frames = total_num_frames
            start = 0
        else:
            sample_frames = self.max_num_frames
            start = np.random.randint(total_num_frames-self.max_num_frames)
        start=start
        stop=sample_frames+start
        if self.ref_id == None:
            ref_id = np.random.randint(total_num_frames)
        elif self.ref_id == "clip":
            ref_id = np.random.randint(sample_frames) + start
        else:
            ref_id = 0
        

        sample_frame_npy = self.get_block_data(path = path, start = start, end = stop)
        sample_hubert_feature_npy = self.get_block_data(path = hubert_path, start = start, end = stop).astype(np.float32)
        sample_lmk_npy = self.get_block_data(path = lmk_path, start = start, end = stop).astype(np.float32)
        sample_pose_list_npy = self.get_block_data(path = pose_path, start = start, end = stop).astype(np.float32)
        sample_eye_blink_list_npy = self.get_block_data(path = eye_blink_path, start = start, end = stop).astype(np.float32)

        ref_frame_npy = self.get_block_data(path = path, start = ref_id, end = ref_id + 1)
        ref_hubert_feature_npy = self.get_block_data(path = hubert_path, start = ref_id, end = ref_id + 1).astype(np.float32)
        ref_pose_list_npy = self.get_block_data(path = pose_path, start = ref_id, end = ref_id + 1).astype(np.float32)
        ref_eye_blink_list_npy = self.get_block_data(path = eye_blink_path, start = ref_id, end =  ref_id + 1).astype(np.float32)

        # mouth_path = os.path.join(self.mouth_dir, video_name+'.npy')
        # mouth_seq = np.load(mouth_path).astype(np.float32)
        # ref_mouth = mouth_seq[ref_id]
        # mouth_seq = mouth_seq[start:stop]
        mouth_lmk_tensor = torch.tensor(sample_lmk_npy[:,48:67])
        

        sample_frame_list = torch.tensor(sample_frame_npy).permute(0,3,1,2)
        sample_hubert_feature_tensor = torch.tensor(sample_hubert_feature_npy)
        sample_frame_list = sample_frame_list - self.mean # 20, 3, 128, 128

        ref_frame_npy = torch.tensor(ref_frame_npy).permute(0,3,1,2)
        ref_hubert_feature_npy = torch.tensor(ref_hubert_feature_npy)
        ref_frame_npy = ref_frame_npy - self.mean # 20, 3, 128, 128

        sample_hubert_feature_tensor = torch.concat([ref_hubert_feature_npy, sample_hubert_feature_tensor], dim = 0)
        sample_frame_list = torch.concat([ref_frame_npy, sample_frame_list], dim = 0)
        # sample_frame_list = [np.transpose(x, (2, 0, 1)) for x in sample_frame_list]
        # sample_frame_list_npy = np.stack(sample_frame_list, axis=1) 
        # sample_pose_list_npy = np.stack(sample_pose_list, axis = 1)
        # sample_eye_blink_list_npy = np.stack(sample_eye_blink_list, axis = 1)
        # change to float32
        sample_frame_list = sample_frame_list.permute(1, 0, 2, 3)
        # sample_frame_list = np.array(sample_frame_list/255.0, dtype=np.float32)  #3, 40, 128, 128
        # sample_frame_list = sample_frame_list/255.  # put to mode l forward
        # added to change the video_name of crema
        video_name = video_name.replace('/','_')

        sample_pose_list_npy = np.concatenate([ref_pose_list_npy, sample_pose_list_npy], axis = 0)
        sample_eye_blink_list_npy = np.concatenate([ref_eye_blink_list_npy, sample_eye_blink_list_npy], axis = 0)
        sample_pose_list_npy = sample_pose_list_npy.transpose(1,0)  # for compatibility
        sample_eye_blink_list_npy = sample_eye_blink_list_npy.transpose(1,0)

        # mouth_seq = np.concatenate([ref_mouth[None], mouth_seq], axis = 0)
        # mouth_seq_npy = mouth_seq.transpose(1,0)
        
        # if __debug__:
        #     end_time = time.time()  # end
        #     print(f'load data time {end_time- start_time}')  # spend lot of time
        #     start_time = end_time
        if self.mode == 'test':
            return sample_frame_list, sample_hubert_feature_tensor, sample_pose_list_npy, sample_eye_blink_list_npy, mouth_lmk_tensor, video_name, start
        return sample_frame_list, sample_hubert_feature_tensor, sample_pose_list_npy, sample_eye_blink_list_npy, mouth_lmk_tensor, video_name, total_num_frames




if __name__ == "__main__":
    # hdtf
    data_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/images_25hz_128_chunk"
    pose_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/pose_bar_chunk"
    eye_blink_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/eye_blink_bbox_from_xpc_bar_2_chunk"
    # crema
    # data_dir='/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/images'
    dataset = HDTF(data_dir=data_dir,
                                       pose_dir=pose_dir,
                                       eye_blink_dir = eye_blink_dir,
                                       image_size=128,
                                       max_num_frames=30,
                                       color_jitter=True)
    for i in range(10):
        dataset.__getitem__(i)
        print('------')    

    test_dataset = data.DataLoader(dataset=dataset,
                                    batch_size=10,
                                    num_workers=8,
                                    shuffle=False)
    for i, batch in enumerate(test_dataset):
        print(i)
