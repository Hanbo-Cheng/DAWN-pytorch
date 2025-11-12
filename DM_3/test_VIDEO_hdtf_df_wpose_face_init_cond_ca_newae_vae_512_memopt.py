# test a video flow diffusion model based on RegionMM for crema dataset\
# 用20帧的ckp一次性动态生成多帧视频，测试集id是和diffused head一样
import sys
sys.path.append('/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw')
import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
from misc import grid2fig, conf2fig
from DM_2.datasets_hdtf_wpose_block_em_lmk_rand_lmdb_test import HDTF
import random
from DM_3.modules.video_flow_diffusion_model_multiGPU_v0_crema_vgg_floss_plus_faceemb_flow_fast_init_cond_rand_vae_2_test import FlowDiffusion
# to save videos
import cv2
import tempfile
from subprocess import call
from pydub import AudioSegment
import re

import gc

import multiprocessing
from multiprocessing import Pool
from functools import partial

import io as sysio
from skimage import io

import lmdb

lmdb_dataset = lmdb.open('/train20/intern/permanent/hbcheng2/data/HDTF/lmdb_HDTF_25hz', readonly=True, lock=False)

start = timeit.default_timer()
BATCH_SIZE = 1
MAX_N_FRAMES = 200
root_dir = '/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw'
# data_dir='/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/images'
data_dir = "/yrfs2/cv2/pcxia/audiovisual/hdtf/images_25hz"
pose_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/pose_bar_chunk"
eye_blink_dir = "/train20/intern/permanent/hbcheng2/data/HDTF/eye_blink_bbox_from_xpc_bar_2_chunk"
GPU = "0"
postfix = "-j-of-tr-ddim"
# default value
ddim_sampling_eta = 1.0
timesteps = 1000
if "ddim" in postfix:
    sampling_step = 20
    ddim_sampling_eta = 1.0
    postfix = postfix + "%04d_%.2f" % (sampling_step, ddim_sampling_eta)
else:
    sampling_step = 1000
INPUT_SIZE = 512
N_FRAMES = 400
RANDOM_SEED = 1234
NUM_VIDEOS = 50
NUM_ITER = NUM_VIDEOS // BATCH_SIZE
MEAN = (0.0, 0.0, 0.0)
cond_scale = 1.0
# the path to trained DM
# RESTORE_FROM = "/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw/data/log_dm_crema/ODOA_v0_Fn=20/snapshots-j-of/flowdiff_0008_S300000.pth"
# RESTORE_FROM = "/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw/data/log_dm_crema/v0_lr_maxFn=20/snapshots-j-of/flowdiff.pth"
# RESTORE_FROM = "/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/data/solve_oom/log_dm_crema/1_newae_v0_Fn=20/snapshots-j-of/flowdiff.pth"
# RESTORE_FROM = '/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/data/origin/log_dm_crema/1_newae_v0_Fn=20/snapshots-j-of/flowdiff.pth'
# RESTORE_FROM = '/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw/data/log_dm_crema/obj_v0_Fn=20/snapshots-j-of/flowdiff_0008_S350000.pth'
# RESTORE_FROM = '/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/data/dfhd_obj_oldAE/log_dm_crema/1_newae_v0_Fn=20/snapshots-j-of/flowdiff_0012_S293000.pth'
# RESTORE_FROM = '/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw/data/log_dm_hdtf/oldae_v0_lr_N=20/snapshots-j-of/flowdiff_copy.pth'
RESTORE_FROM = '/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/data/HDTF_wpose_faceemb_512_vae/acc_cross_attn_vae_KL_1e-4_extend_ft_4/df_ref0_oldae_v0_lr_N=10/snapshots-j-of-rf/flowdiff_0014_S217500.pth'
# RESTORE_FROM = '/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw/data/log_dm_hdtf/start0_v0_lr_N=10/snapshots-j-of/flowdiff.pth'
name = 'DF_' + str(MAX_N_FRAMES) +'_'+ RESTORE_FROM.split('/')[9] + '_' + RESTORE_FROM.split('/')[11].split('.')[0]+'_ddim'+str(sampling_step) + '_df'
# name = 'DF_bjd_' + RESTORE_FROM.split('/')[8] + '_' + RESTORE_FROM.split('/')[12].split('.')[0] 
AE_RESTORE_FROM = '/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/AE/data/log-hdtf-512-cosin/hdtf512_400ep_2024-11-03_20:43/snapshots/RegionMM.pth'   
VAE_RESTORE_FROM = '/train20/intern/permanent/hbcheng2/AIGC_related/DAWN-vqvae/AE/data/log-hdtf-512-compress-KL-sd/hdtf512_100ep_2025-01-15_11:20_0.0001/snapshots/VAE_0020_S410000.pth'
# AE_RESTORE_FROM = '/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/AE/data/log-hdtf/hdtf128_llm_2024-07-26_12:54/snapshots/RegionMM.pth'
# AE_RESTORE_FROM = "/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw/data/log-hdtf/hdtf128_2024-01-06_21:35/snapshots/RegionMM.pth" # old AE for crema
# AE_RESTORE_FROM = '/train20/intern/permanent/lmlin2/Flow/CVPR23_LFDM_sw/data/log-hdtf/hdtf128_2024-02-11_15:45/snapshots/RegionMM_0100_S074360.pth'  # new AE for crema
# CKPT_DIR = os.path.join(root_dir, 'demo','test_ndna30w_-1_notall_fstart_reg_v0_19'+postfix)

DATASAVE_DIR = '/train20/intern/permanent/hbcheng2/data'
CKPT_DIR = os.path.join(DATASAVE_DIR, 'hdtf_wpose_faceemb_all_newae_vae_512','tmp', name ,'video')
os.makedirs(CKPT_DIR, exist_ok=True)
IMG_DIR = os.path.join(DATASAVE_DIR, 'hdtf_wpose_faceemb_all_newae_vae_512','tmp' , name ,'img')
os.makedirs(IMG_DIR, exist_ok=True)
print(root_dir)
print(DATASAVE_DIR)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("cond scale:", cond_scale)
print("sampling step:", sampling_step)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--num-workers", default=4)
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
    return parser.parse_args()


args = get_arguments()



def extract_audio_by_frames(input_wav_path, start_frame_index, num_frames, frame_rate, output_wav_path):
    # 读取音频文件
    audio = AudioSegment.from_wav(input_wav_path)

    # 计算每一帧的时间间隔
    frame_duration = 1000 / frame_rate  # 假设帧率是每秒的帧数

    # 计算选择的帧对应的时间范围
    start_time_ms = start_frame_index * frame_duration
    end_time_ms = (start_frame_index + num_frames) * frame_duration

    # 切割选择的时间范围
    selected_audio = audio[start_time_ms:end_time_ms]

    # 保存提取的音频
    selected_audio.export(output_wav_path, format="wav")

def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)

def inference_one_segment(model, 
                          real_vids, 
                          ref_hubert, 
                          real_poses, 
                          real_blink_bbox, 
                          real_mouth_ratio, 
                          start_frame_index,
                          real_names,
                          i_iter,
                          segment_id,
                          total_start):

    start_frame_index = start_frame_index + total_start
    bs = real_vids.size(0)
    fn = real_vids.size(2)

    ref_imgs = real_vids[:, :, 0, :, :].clone().detach()

    reg_count = 0

    tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir='/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/demo')
    output_wav_path = tempfile.NamedTemporaryFile('w', suffix='.wav', dir='/train20/intern/permanent/hbcheng2/AIGC_related/CVPR23_LFDM_sw/demo').name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25  # 帧率

    # SAV_DIR = os.path.join(CKPT_DIR, str(i_iter)+'_'+real_names[0]+'.mp4') # TODO: 保存路径需要修改
    SAV_DIR = os.path.join(CKPT_DIR, str(i_iter)+'_'+real_names[0],  '%03d_%05d_%04d.mp4' % (segment_id, start_frame_index, fn))

    # if os.path.exists(SAV_DIR):
    #     return 
    # 创建视频写入对象
    video_writer = cv2.VideoWriter(tmp_video_file_pred.name, fourcc, fps, (512, 512))

    wav_path = os.path.join("/yrfs2/cv2/pcxia/audiovisual/hdtf/images_25hz".replace('/images_25hz','/image_audio'), real_names[0]+'.wav')
    # wav_path = os.path.join(testloader.dataset.data_dir.replace('/images','/audio'), real_names[0].replace('_','/',1)+'.wav')
    # start_frame_index = 0
    extract_audio_by_frames(wav_path, start_frame_index, fn, fps, output_wav_path)

    # 创建imgs文件夹
    img_dir_name = "%04d_%s_%.2f" % (i_iter, real_names[0], cond_scale)
    cur_img_dir_gt = os.path.join(IMG_DIR, img_dir_name,'gt', '%03d_%05d_%04d' % (segment_id, start_frame_index, fn))  # TODO: 保存路径需要修改
    os.makedirs(cur_img_dir_gt, exist_ok=True)
    cur_img_dir_samp = os.path.join(IMG_DIR, img_dir_name,'samp', '%03d_%05d_%04d' % (segment_id, start_frame_index, fn))  # TODO: 保存路径需要修改
    os.makedirs(cur_img_dir_samp, exist_ok=True)

    with torch.no_grad():
        model.update_num_frames(fn)
        # cond = torch.concat([ref_hubert[0].unsqueeze(dim=0), real_poses[0].permute(1,0).unsqueeze(0), real_blink_bbox[0][:2].permute(1,0).unsqueeze(0)], dim=-1).cuda()  # manually concat pose
        # sample_output_dict = model.sample_one_video(sample_img=ref_imgs.cuda(),
        #                                             sample_audio_hubert=cond,
        #                                             sample_bbox = real_blink_bbox[0][2:].unsqueeze(dim=0).cuda(),
        #                                             cond_scale=1.0)
        sample_output_dict = model.sample_one_video(real_vid=real_vids.cuda()/255.,
                                                                    sample_img=ref_imgs[0].unsqueeze(dim=0).cuda()/255.,
                                                                    sample_audio_hubert = ref_hubert[0].unsqueeze(dim=0).cuda(),
                                                                    sample_pose = real_poses[0].unsqueeze(0).cuda(),
                                                                    sample_eye =  real_blink_bbox[0][:2].unsqueeze(0).cuda(),
                                                                    sample_bbox = real_blink_bbox[0,2:].unsqueeze(0).cuda(),
                                                                    cond_scale=1.0)
    
    msk_size = INPUT_SIZE
        
    for batch_idx in range(bs):

        for frame_idx in range(0,fn):

            new_im_gt = Image.new('RGB', (msk_size, msk_size))
            new_im_sample = Image.new('RGB', (msk_size, msk_size))

            save_sample_out_img = sample_img(sample_output_dict["sample_out_vid"][:, :, frame_idx], batch_idx)
            save_tar_img = sample_img(real_vids[:, :, reg_count+frame_idx]/255., batch_idx)
            # save_real_out_img = sample_img(train_output_dict["real_out_vid"][:, :, frame_idx], batch_idx)

            # save sample videos
            frame_rgb = np.uint8(save_sample_out_img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            # save sample and gt imgs
            new_im_gt.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
            new_im_sample.paste(Image.fromarray(save_sample_out_img, 'RGB'), (0, 0))
            new_im_arr_gt = np.array(new_im_gt)
            new_im_arr_sample = np.array(new_im_sample)
            new_im_name = "%03d_%s_%.2f.png" % (frame_idx+reg_count, real_names[batch_idx], cond_scale)
            imageio.imsave(os.path.join(cur_img_dir_gt,new_im_name), new_im_arr_gt)
            imageio.imsave(os.path.join(cur_img_dir_samp,new_im_name), new_im_arr_sample)

            del new_im_sample, new_im_arr_sample, new_im_arr_gt, frame_rgb, frame_bgr, save_tar_img, save_sample_out_img, new_im_gt
        
    video_writer.release()
    cmd = ('ffmpeg -y ' + ' -i {0} -i {1} -vcodec copy -ac 2 -channel_layout stereo -pix_fmt yuv420p {2} -shortest'.format(
    output_wav_path, tmp_video_file_pred.name, SAV_DIR)).split()   
    call(cmd)  
    iter_end = timeit.default_timer()

    try:
        os.remove(tmp_video_file_pred.name)
        os.remove(output_wav_path)
    except OSError as e:
        print(f'Error: {e.strerror}')

    del sample_output_dict
    del ref_imgs
    gc.collect()

    # pass

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

def load_image(path):
        """加载图像并将其转换为numpy数组"""
        resize_fn = partial(resize, desired_size=INPUT_SIZE, interpolation=cv2.INTER_AREA)
        global lmdb_dataset
        with lmdb_dataset.begin() as txn:
            video_array = resize_fn(io.imread( sysio.BytesIO(txn.get(path.encode('utf-8')))) )
        return np.array(video_array)

     
def load_image_with_index(index, path):
        return index, load_image(path[0])

def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowDiffusion(is_train=True,
                          img_size=INPUT_SIZE //32,
                          sampling_timesteps=sampling_step,
                          ddim_sampling_eta=ddim_sampling_eta,
                          num_frames=N_FRAMES,
                          config_pth="/train20/intern/permanent/hbcheng2/AIGC_related/DAWN-vqvae/AE/data/log-hdtf-512-compress-KL-sd/hdtf512_100ep_2025-01-15_11:20_0.0001/hdtf512_100ep.yaml",
                          pretrained_pth=AE_RESTORE_FROM,
                          vae_ckpt=VAE_RESTORE_FROM,)
                          # vae_config = '/train20/intern/permanent/hbcheng2/AIGC_related/DAWN-vqvae/AE/data/log-hdtf-512-compress-KL-sd/hdtf512_100ep_2025-01-15_11:20_0.0001')
    

    if args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from, map_location = 'cpu')
            model.diffusion.load_state_dict(checkpoint['diffusion'])
            model.image_projector.load_state_dict(checkpoint['image_projector'])
            model.face_loc_emb.load_state_dict(checkpoint['face_loc_emb'])
            print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
            exit(-1)
    else:
        print("NO checkpoint found!")
        exit(-1)
    model.cuda()
    model.eval()

    setup_seed(args.random_seed)
    dataset = HDTF( data_dir=data_dir,
                    pose_dir=pose_dir,
                    eye_blink_dir = eye_blink_dir,
                    image_size=INPUT_SIZE,
                    mode='test',
                    # audio_dir='/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wavlm_interpolate_chunk',
                    max_num_frames=1e8,
                    color_jitter=True,
                    mean=MEAN)
    testloader = data.DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True, num_workers=1,
                                 pin_memory=False)



    

    def load_images_in_order(image_paths):
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            # 使用 partial 将索引和路径传递给 load_image_with_index 函数
            results = pool.starmap(load_image_with_index, (
            (i, path) for i, path in enumerate(image_paths)
        ))
        
        # 按照索引排序结果
        results.sort(key=lambda x: x[0])
        
        # 提取图像
        images = [image for index, image in results]
        images = np.stack(images, axis=0)
        
        return images

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()
    
    
    for i_iter, batch in enumerate(testloader):

        reg_count = 0

        if i_iter >= NUM_ITER:
            break
        data_time.update(timeit.default_timer() - iter_end)
        real_vids, ref_hubert, real_poses, real_blink_bbox, mouth_lmk_tensor, _, real_names, start_frame_index = batch
        
        batch_time.update(timeit.default_timer() - iter_end)

        # with torch.no_grad():
        #     train_output_dict = model.forward(real_vid=real_vids_.cuda(), ref_img=ref_imgs.cuda(), ref_text=ref_hubert_.cuda())

        fn = len(real_vids)

        seg_id = 0

        video_dir = os.path.join(CKPT_DIR, str(i_iter)+'_'+real_names[0])

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        early_break = False
        for i in range(0, fn, MAX_N_FRAMES):
            

            start_frame = i
            end_frame = min(fn, i + MAX_N_FRAMES)

            if fn - end_frame < 100:
                end_frame = fn
                early_break = True

            SAV_DIR = os.path.join(CKPT_DIR, str(i_iter)+'_'+real_names[0],  '%03d_%05d_%04d.mp4' % (seg_id, start_frame_index, end_frame - start_frame))
            if os.path.exists(SAV_DIR):
                seg_id += 1
                continue
            seg_real_vids = real_vids[start_frame: end_frame]
            seg_ref_hubert = ref_hubert[:, start_frame: end_frame] # ref_hubert: b, fn, c
            seg_real_poses = real_poses[:, :, start_frame: end_frame] # ref_hubert: b, c, fn
            seg_real_blink_bbox = real_blink_bbox[:, :, start_frame: end_frame]
            # seg_real_mouth_ratio = real_mouth_ratio[:, start_frame: end_frame]
            
            seg_real_vids = load_images_in_order(seg_real_vids)
            seg_real_vids = torch.tensor(seg_real_vids).permute(0,3,1,2)
            seg_real_vids = seg_real_vids.permute(1, 0, 2, 3).unsqueeze(0) # b, 3, nf, H, W
            inference_one_segment(model = model, 
                          real_vids = seg_real_vids, 
                          ref_hubert = seg_ref_hubert, 
                          real_poses = seg_real_poses, 
                          real_blink_bbox = seg_real_blink_bbox, 
                          real_mouth_ratio = None, 
                          start_frame_index = start_frame,
                          real_names = real_names,
                          i_iter = i_iter,
                          segment_id = seg_id,
                          total_start = start_frame_index)
            
            seg_id += 1
            if early_break:
                break
            # break

        if i_iter % args.print_freq == 0:
            print('Test:[{0}/{1}]\t'
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                  .format(i_iter, NUM_ITER, batch_time=batch_time))

    end = timeit.default_timer()
    print(end - start, 'seconds')
    print(CKPT_DIR)
    # print(IMG_DIR)


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

