import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(parent_dir)

import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn

import timeit
from PIL import Image
# from misc import grid2fig, conf2fig
import random
from DM_3.modules.video_flow_diffusion_model_multiGPU_v0_crema_vgg_floss_plus_faceemb_flow_fast_init_cond_test import FlowDiffusion
# to save videos
import cv2
import tempfile
from subprocess import call
from pydub import AudioSegment
import re
from torchvision import transforms
import time
import datetime
BATCH_SIZE = 1
MAX_N_FRAMES = 200
GPU = "0"
INPUT_SIZE = 128
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
WIN_WIDTH = 40
postfix = "-j-of-tr-ddim"
RESTORE_FROM = '.\pretrain_models\DAWN_128.pth'
# RESTORE_FROM = 'your_path/data/log_dm_hdtf/start0_v0_lr_N=10/snapshots-j-of/flowdiff.pth'
# name = 'DF_' + str(MAX_N_FRAMES) +'_'+ RESTORE_FROM.split('/')[9] + '_' + RESTORE_FROM.split('/')[11].split('.')[0]+'_ddim'+str(sampling_step)
# name = 'DF_bjd_' + RESTORE_FROM.split('/')[8] + '_' + RESTORE_FROM.split('/')[12].split('.')[0]



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--source_img_path", type=str)
    parser.add_argument("--init_state_path", type=str,  help="dir, init state path")
    parser.add_argument("--drive_blink_path", type=str,  help="npyfile, drive blink path, 25hz")
    parser.add_argument("--drive_pose_path", type=str,  help="npyfile, drive pose path, 25hz")
    parser.add_argument("--audio_emb_path", type=str,  help="npy file, hubert embedding, 25hz")
    parser.add_argument("--src_audio_path", type=str,  help="wav file, source audio")
    parser.add_argument("--save_path", type=str,  help="save path")
    parser.add_argument("--max_n_frames", type=int, default=MAX_N_FRAMES, help="save path")
    return parser.parse_args()


args = get_arguments()

if args.input_size == 256:
    AE_RESTORE_FROM = '.\pretrain_models\LFG_256_400ep.pth'
else:
    AE_RESTORE_FROM = '.\pretrain_models\LFG_128_1000ep.pth' 
start = timeit.default_timer()

# default value
ddim_sampling_eta = 1.0
POSE_DIM = 6 # 7
timesteps = 1000
if "ddim" in postfix:
    sampling_step = 20
    ddim_sampling_eta = 1.0
    postfix = postfix + "%04d_%.2f" % (sampling_step, ddim_sampling_eta)
else:
    sampling_step = 1000

cond_scale = 1.0


directory_name = (args.source_img_path).split('/')[-1].split('.')[0]
CKPT_DIR = os.path.join(args.save_path, directory_name ,'video')
os.makedirs(CKPT_DIR, exist_ok=True)
IMG_DIR = os.path.join(args.save_path, directory_name, 'img')
os.makedirs(IMG_DIR, exist_ok=True)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("cond scale:", cond_scale)
print("sampling step:", sampling_step)




def extract_audio_by_frames(input_wav_path, start_frame_index, num_frames, frame_rate, output_wav_path):
    audio = AudioSegment.from_wav(input_wav_path)
    frame_duration = 1000 / frame_rate
    start_time_ms = start_frame_index * frame_duration
    end_time_ms = (start_frame_index + num_frames) * frame_duration
    selected_audio = audio[start_time_ms:end_time_ms]
    selected_audio.export(output_wav_path, format="wav")

def get_block_data(path, start, end):
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

def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)

def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowDiffusion(is_train=True,
                          sampling_timesteps=sampling_step,
                          ddim_sampling_eta=ddim_sampling_eta,
                          pose_dim = POSE_DIM,
                          config_pth=f".\config\hdtf{str(args.input_size)}.yaml",
                          pretrained_pth=AE_RESTORE_FROM,
                          win_width = WIN_WIDTH)
    model.cuda()

    if args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            model.diffusion.load_state_dict(checkpoint['diffusion'])
            print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
            exit(-1)
    else:
        print("NO checkpoint found!")
        exit(-1)

    model.eval()


    drive_face_id = 'WRA_JohnKasich1_001'
    ref_hubert_path = args.audio_emb_path
    drive_pose_path = args.drive_pose_path
    real_pose_path = os.path.join(args.init_state_path, 'init_pose.npy')
    ref_img_path = args.source_img_path
    real_blink_bbox_path = os.path.join(args.init_state_path, 'init_eye_bbox.npy')
    drive_blink_path = args.drive_blink_path

    ref_id = 0
    image = Image.open(ref_img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image) * 255
    hubert_npy = np.load(ref_hubert_path)
    # MAX_N_FRAMES
    MAX_N_FRAMES = min(args.max_n_frames, hubert_npy.shape[0])
    ref_hubert = torch.from_numpy(hubert_npy[:MAX_N_FRAMES]).to(torch.float32)
    drive_poses = torch.from_numpy(np.load(drive_pose_path)[:MAX_N_FRAMES]).to(torch.float32)
    drive_blink_bbox = torch.from_numpy(np.load(drive_blink_path)[:MAX_N_FRAMES]).to(torch.float32)
    
    try:
        real_poses = torch.from_numpy(np.load(real_pose_path)[:MAX_N_FRAMES]).to(torch.float32)
        real_blink_bbox = torch.from_numpy(np.load(real_blink_bbox_path)[:MAX_N_FRAMES]).to(torch.float32)
    except Exception:
        real_poses = torch.zeros(1 , 6)
        real_blink_bbox = torch.tensor([[0.3, 0.3, 64 , 64, 192, 192, 256, 256]]).reshape(1, -1).to(torch.float32)
        print(real_poses.shape)
        print(real_blink_bbox.shape)
    init_pose = real_poses[ref_id].unsqueeze(0)
    init_blink = real_blink_bbox[ref_id,:2].unsqueeze(0)

    real_poses = real_poses.permute(1,0)
    drive_poses = drive_poses.permute(1,0)
    drive_blink_bbox = drive_blink_bbox.permute(1,0)
    real_blink_bbox = real_blink_bbox.permute(1,0)


    real_names = ref_hubert_path.split('/')[-1][:-4] # 'WRA_JoeHeck1_000'
    setup_seed(args.random_seed)
    
    output_wav_path = tempfile.NamedTemporaryFile('w', suffix='.wav', dir='./', delete=False)

    reg_count = 0

    fn = MAX_N_FRAMES

    SAV_DIR = os.path.join(CKPT_DIR,  real_names+'.mp4')
    # if os.path.exists(SAV_DIR):
    #     continue

    tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir='./', delete=False)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25 

    video_writer = cv2.VideoWriter(tmp_video_file_pred.name, fourcc, fps, (args.input_size, args.input_size))

    extract_audio_by_frames(args.src_audio_path, 0, fn, fps, output_wav_path.name)

    img_dir_name = "%s_%.2f" % (real_names, cond_scale)
    cur_img_dir_gt = os.path.join(IMG_DIR, img_dir_name,'gt')
    os.makedirs(cur_img_dir_gt, exist_ok=True)
    cur_img_dir_samp = os.path.join(IMG_DIR, img_dir_name,'samp')
    os.makedirs(cur_img_dir_samp, exist_ok=True)


    # ref_imgs =( real_vids[:, :, 0, :, :].clone().detach())
    bs = 1
    for i in range(1):
        start_time = time.time()  # end
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                model.update_num_frames(fn)
                # cond = torch.concat([ref_hubert[0].unsqueeze(dim=0), real_poses[0].permute(1,0).unsqueeze(0), real_blink_bbox[0][:2].permute(1,0).unsqueeze(0)], dim=-1).cuda()  # manually concat pose
                sample_output_dict = model.sample_one_video(sample_img=image_tensor.unsqueeze(dim=0).cuda()/255.,
                                                            sample_audio_hubert = ref_hubert.unsqueeze(dim=0).cuda(),
                                                            sample_pose = drive_poses.unsqueeze(0).cuda(),
                                                            sample_eye =  drive_blink_bbox[:2].unsqueeze(0).cuda(),
                                                            sample_bbox = real_blink_bbox[2:].unsqueeze(0).cuda(),
                                                            init_pose = init_pose.cuda(),
                                                            init_eye = init_blink.cuda(),
                                                            cond_scale=1.0)
        end_time = time.time()  # end
        print(f'generation time {end_time- start_time}')
        start_time = end_time
    
    msk_size = args.input_size
    
    for batch_idx in range(bs):

        for frame_idx in range(0,fn):
            new_im_sample = Image.new('RGB', (msk_size, msk_size))

            save_sample_out_img = sample_img(sample_output_dict["sample_out_vid"][:, :, frame_idx], batch_idx)

            # save sample videos
            frame_rgb = np.uint8(save_sample_out_img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            # save sample imgs
            new_im_sample.paste(Image.fromarray(save_sample_out_img, 'RGB'), (0, 0))
            new_im_arr_sample = np.array(new_im_sample)
            new_im_name = "%03d_%s_%.2f.png" % (frame_idx+reg_count, real_names[batch_idx], cond_scale)
            imageio.imsave(os.path.join(cur_img_dir_samp,new_im_name), new_im_arr_sample)
        
    video_writer.release()
    cmd = ('ffmpeg -y ' + ' -i {0} -i {1} -vcodec copy -ac 2 -channel_layout stereo -pix_fmt yuv420p {2} -shortest'.format(
    output_wav_path.name, tmp_video_file_pred.name, SAV_DIR)).split()   
    call(cmd)  
    delete_file(output_wav_path.name)
    delete_file(tmp_video_file_pred.name)
    iter_end = timeit.default_timer()


    end = timeit.default_timer()
    print(end - start, 'seconds')
    print(CKPT_DIR)
    # print(IMG_DIR)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import os

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} has been deleted successfully.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except PermissionError:
        print(f"Permission denied: Unable to delete {file_path}.")
    except Exception as e:
        print(f"Error occurred while deleting {file_path}: {e}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()

