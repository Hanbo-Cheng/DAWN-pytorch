# build MUG dataset for RegionMM

import os
import imageio

import numpy as np
from torch.utils.data import Dataset
import yaml
from argparse import ArgumentParser
from augmentation import AllAugmentationTransform

from functools import partial
import cv2
import matplotlib.pyplot as plt
import imageio.v2 as imageio


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


# this is just for training
class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=256, id_sampling=False,
                 pairs_list=None, augmentation_params=None):
        
        self.root_dir = root_dir
        self.frame_shape = frame_shape
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling


        
        vid_list = []
        # crema
        for id_name in os.listdir(root_dir):
            vid_list.extend([os.path.join(id_name, sent) for sent in os.listdir(f'{root_dir}/{id_name}') ])

         #hdtf
        # for id_name in os.listdir(root_dir):
        #     vid_list.append(id_name)
            
        self.videos = vid_list

        self.transform = AllAugmentationTransform(**augmentation_params)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.id_sampling:
            raise NotImplementedError
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        frames = os.listdir(path)
        frames.sort()

        num_frames = len(frames)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
        resize_fn = partial(resize, desired_size=self.frame_shape, interpolation=cv2.INTER_AREA)

        if type(frames[0]) is bytes:
            frame_names = [frames[idx].decode('utf-8') for idx in frame_idx]
        else:
            frame_names = [frames[idx] for idx in frame_idx]

        video_array = [resize_fn(imageio.imread(os.path.join(path, x))) for x in frame_names]

        # video_array = [img_as_float32(x) for x in video_array]

        video_array = self.transform(video_array)

        out = {}

        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')

        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['name'] = name
        out['frame'] = frame_names
        out['id'] = idx
        
        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config",
                        default="/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM-main/config/hdtf128.yaml",
                        help="path to config")
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    data = FramesDataset(**config['dataset_params'])

    # data.__getitem__(0)

    # print('_------')
    # data.__getitem__(1)
    # print('------')
    data.__getitem__(2)
    print('------')    
    data.__getitem__(3)
    print('------')