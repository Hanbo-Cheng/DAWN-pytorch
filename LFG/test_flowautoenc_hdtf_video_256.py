# use LFG to reconstruct testing videos and measure the loss in video domain
# using RegionMM

import argparse
import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
import sys
sys.path.append("your/path/DAWN-pytorch")
from misc import grid2fig
from DM.datasets_hdtf_wpose_lmk_mo_block_mraa import HDTF_test as HDTF
import random
from LFG.modules.flow_autoenc import FlowAE
import torch.nn.functional as F
from LFG.modules.util import Visualizer
import json_tricks as json
import cv2
import tempfile
from subprocess import call
from pydub import AudioSegment
from einops import rearrange
from tqdm import tqdm

start = timeit.default_timer()
BATCH_SIZE = 1
INPUT_SIZE = 256
root_dir = 'your/path/DAWN-pytorch/AE'  # your work directory
data_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/images_25hz_256_chunk"
pose_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/pose_bar_chunk"
eye_blink_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/eye_blink_bbox_from_xpc_bar_2_chunk"

DATASAVE_DIR = '/train20/intern/permanent/hbcheng2/data'
CKPT_DIR = os.path.join(DATASAVE_DIR, 'mraa_result', str(INPUT_SIZE) + '_400ep','video')
os.makedirs(CKPT_DIR, exist_ok=True)
IMG_DIR = os.path.join(DATASAVE_DIR, 'mraa_result', str(INPUT_SIZE) + '_400ep','img')
os.makedirs(IMG_DIR, exist_ok=True)

# GPU = "6"
postfix = ""

N_FRAMES = 40
NUM_VIDEOS = 10
SAVE_VIDEO = True
NUM_ITER = NUM_VIDEOS // BATCH_SIZE
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
# the path to trained LFG model
RESTORE_FROM ='your/path/DAWN-pytorch/AE/data/log-hdtf-256-cosin/hdtf256_400ep_2024-08-08_00:15/snapshots/RegionMM.pth'
# RESTORE_FROM = "/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM-main/data/log-hdtf/hdtf256_2023-11-21_16:49/snapshots/RegionMM_0020_S080000.pth"
config_pth = "your/path/DAWN-pytorch/config/hdtf256.yaml"

json_path = os.path.join(CKPT_DIR, "loss%d%s.json" % (NUM_VIDEOS, postfix))
visualizer = Visualizer()
print(root_dir)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("config_path:", config_pth)
print(json_path)
print("save video:", SAVE_VIDEO)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Autoencoder")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--gpu", default=0,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()


args = get_arguments()

def extract_audio_by_frames(input_wav_path, start_frame_index, num_frames, frame_rate, output_wav_path):
    # 
    audio = AudioSegment.from_wav(input_wav_path)

    # 
    frame_duration = 1000 / frame_rate  # 

    # 
    start_time_ms = start_frame_index * frame_duration
    end_time_ms = (start_frame_index + num_frames) * frame_duration

    # 
    selected_audio = audio[start_time_ms:end_time_ms]

    # 
    selected_audio.export(output_wav_path, format="wav")



def sample_img(rec_img_batch):
    rec_img = rec_img_batch.permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowAE(is_train=False, config_pth=config_pth)
    model.cuda()

    if os.path.isfile(args.restore_from):
        print("=> loading checkpoint '{}'".format(args.restore_from))
        checkpoint = torch.load(args.restore_from)
        model.generator.load_state_dict(checkpoint['generator'])
        model.region_predictor.load_state_dict(checkpoint['region_predictor'])
        model.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        print("=> loaded checkpoint '{}'".format(args.restore_from))
    else:
        print("=> no checkpoint found at '{}'".format(args.restore_from))
        exit(-1)

    model.eval()

    setup_seed(args.random_seed)

    testloader = data.DataLoader(HDTF(data_dir=data_dir,
                                       pose_dir=pose_dir,
                                       eye_blink_dir = eye_blink_dir,
                                       image_size=INPUT_SIZE,
                                       mode='test',
                                       max_num_frames=1e8,
                                       color_jitter=True,
                                       mean=MEAN),
                                 batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=8,
                                 pin_memory=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()
    cnt = 0

    out_loss = 0.0
    warp_loss = 0.0
    num_sample = 0.0
    l1_loss = torch.nn.L1Loss(reduction='sum')

    global_iter = 0


    while global_iter < NUM_ITER:
        for i_iter, batch in enumerate(testloader):
            # if i_iter < NUM_ITER:
            #     break
            # if global_iter < NUM_ITER:
            #     break

            data_time.update(timeit.default_timer() - iter_end)

            block_path_list, real_names, total_num_frames = batch

            out_img_list = []
            # use first frame of each video as reference frame

            ref_path = block_path_list[0][0]
            ref_imgs = np.load(ref_path) # 25, 256, 256, 3
            ref_imgs = torch.tensor(ref_imgs).permute(0, 3, 1, 2)
            ref_imgs = ref_imgs[0].clone().detach().to(torch.float32)/255.
            ref_img_tmp = ref_imgs.repeat(25,1,1,1).reshape(-1, 3, INPUT_SIZE, INPUT_SIZE)
            
            msk_size = ref_imgs.shape[-1]
            new_im_list = []
            img_dir_name = "%04d_%s" % (i_iter, real_names[0])
            cur_img_dir_gt = os.path.join(IMG_DIR, img_dir_name,'gt')
            os.makedirs(cur_img_dir_gt, exist_ok=True)
            cur_img_dir_samp = os.path.join(IMG_DIR, img_dir_name,'mraa') 
            os.makedirs(cur_img_dir_samp, exist_ok=True)
            
            fps = 25  # 

            tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir='your/path/DAWN-pytorch/demo')
            output_wav_path = tempfile.NamedTemporaryFile('w', suffix='.wav', dir='your/path/DAWN-pytorch/demo').name

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(tmp_video_file_pred.name, fourcc, fps, (INPUT_SIZE, INPUT_SIZE))
            SAV_DIR = os.path.join(CKPT_DIR, str(i_iter)+'_'+real_names[0] + '.mp4')


            wav_path = os.path.join("/yrfs2/cv2/pcxia/audiovisual/hdtf/images_25hz".replace('/images_25hz','/image_audio'), real_names[0]+'.wav')

            extract_audio_by_frames(wav_path, 0, total_num_frames, fps, output_wav_path)

            batch_time.update(timeit.default_timer() - iter_end)

            new_im_gt = Image.new('RGB', (msk_size, msk_size))
            new_im_sample = Image.new('RGB', (msk_size, msk_size))

            frame_cnt = 0
            for id in range(len(block_path_list)):
                block_path = block_path_list[id][0]
                real_vids = np.load(block_path)
                if real_vids.shape[0] !=ref_img_tmp.shape[0]:
                    ref_img_tmp = ref_imgs.repeat(real_vids.shape[0],1,1,1).reshape(-1, 3, INPUT_SIZE, INPUT_SIZE)
                real_vids = torch.tensor(real_vids).permute(0, 3, 1, 2)  # 25, 256, 256, 3 - > 25, 3, 256, 256

                

                dri_imgs = real_vids.to(torch.float32)/255.
                with torch.no_grad():
                    model.set_train_input(ref_img=ref_img_tmp, dri_img=dri_imgs)
                    model.forward()
                out_img_tensor = (model.generated['prediction'] * 255.).to(torch.uint8).clone().detach().cpu()

                # save real_vids
                for i in range(real_vids.shape[0]):
                    save_tar_img = np.array(real_vids[i].permute(1, 2, 0), np.uint8)
                    save_out_img = np.array(out_img_tensor[i].permute(1, 2, 0), np.uint8)
                    frame_rgb = np.uint8(save_out_img)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

                    new_im_gt.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
                    new_im_sample.paste(Image.fromarray(save_out_img, 'RGB'), (0, 0))
                    new_im_arr_gt = np.array(new_im_gt)
                    new_im_arr_sample = np.array(new_im_sample)
                    new_im_name = "%03d_%s.png" % (frame_cnt, real_names[0])
                    imageio.imsave(os.path.join(cur_img_dir_gt,new_im_name), new_im_arr_gt)
                    imageio.imsave(os.path.join(cur_img_dir_samp,new_im_name), new_im_arr_sample)

                    frame_cnt += 1


            # out_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), out_img_list_tensor.cpu()).item()
            # warp_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), warped_img_list_tensor.cpu()).item()
            
            
            
            video_writer.release()
            cmd = ('ffmpeg -y ' + ' -i {0} -i {1} -vcodec copy -ac 2 -channel_layout stereo -pix_fmt yuv420p {2} -shortest'.format(
            output_wav_path, tmp_video_file_pred.name, SAV_DIR)).split()  
                
            call(cmd)  
            try:
                os.remove(tmp_video_file_pred.name)
                os.remove(output_wav_path)
            except OSError as e:
                print(f'Error: {e.strerror}')

            iter_end = timeit.default_timer()

            if global_iter % args.print_freq == 0:
                print('Test:[{0}/{1}]\t'
                      'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                      .format(global_iter, NUM_ITER, batch_time=batch_time))
            global_iter += 1

    print("loss for prediction: %.5f" % (out_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)))
    print("loss for warping: %.5f" % (warp_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)))

    res_dict = {}
    res_dict["out_loss"] = out_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)
    res_dict["warp_loss"] = warp_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)
    with open(json_path, "w") as f:
        json.dump(res_dict, f)

    end = timeit.default_timer()
    print(end - start, 'seconds')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
