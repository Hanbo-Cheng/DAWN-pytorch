import os
import os.path as osp
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.append('.')  # 添加当前目录到Python路径

import os
import cv2
import yaml
import tempfile
import numpy as np
import torch

import soundfile as sf
from scipy.interpolate import interp1d
from extract_init_states.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from extract_init_states.TDDFA_ONNX import TDDFA_ONNX
from extract_init_states.utils.pose import get_pose
from extract_init_states.utils.functions import calculate_eye, calculate_bbox

from transformers import AutoProcessor, HubertModel

from PBnet.src.models.get_model import get_model as get_gen_model
from PIL import Image
from torchvision import transforms

from pydub import AudioSegment

def inv_transform(x, min_vals, max_vals):
    """反归一化函数"""
    return x * (max_vals - min_vals) + min_vals

def load_args(filename):
    with open(filename, "rb") as optfile:
        opt = yaml.load(optfile, Loader=yaml.Loader)
    return opt

class VideoGenerator:
    def __init__(self, args):
        self.audio_path = args.audio_path
        self.image_path = args.image_path
        self.output_path = args.output_path
        self.cache_path = args.cache_path
        
        # 确保输出目录存在 
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        
        # 设置中间文件路径
        self.audio_emb_path = os.path.join(self.cache_path, 'target_audio.npy')

        # 设置ONNX运行环境 3DDFA
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '8'
        
        # 初始化配置
        self.config_path = './extract_init_states/configs/mb1_120x120.yml'
        self.cfg = yaml.load(open(self.config_path), Loader=yaml.SafeLoader)
        
        # 初始化模型
        self.face_boxes = FaceBoxes_ONNX()
        self.tddfa = TDDFA_ONNX(**self.cfg)

        # HuBERT模型相关配置
        print("Loading the Wav2Vec2 Processor...")
        self.wav2vec2_processor = AutoProcessor.from_pretrained("/disk2/pfhu/hubert-large-ls960-ft")
        print("Loading the HuBERT Model...")
        self.hubert_model = HubertModel.from_pretrained("/disk2/pfhu/hubert-large-ls960-ft")
        self.hubert_model.eval()
        # 添加PBnet相关配置
        self.pbnet_pose_ckpt = './pretrain_models/pbnet_seperate/pose/checkpoint_40000.pth.tar'
        self.pbnet_blink_ckpt = './pretrain_models/pbnet_seperate/blink/checkpoint_95000.pth.tar'
        self.device = 'cuda:0'
        
        # PBnet模型参数
        folder_p, _ = os.path.split(self.pbnet_pose_ckpt)
        self.pose_params = load_args(os.path.join(folder_p, "opt.yaml"))
        self.pose_params['device'] = self.device
        self.pose_params['audio_dim'] = 1024
        self.pose_params['pos_dim'] = 6
        self.pose_params['eye_dim'] = 0
        
        
        folder_b, _ = os.path.split(self.pbnet_blink_ckpt)
        self.blink_params = load_args(os.path.join(folder_b, "opt.yaml"))
        self.blink_params['device'] = self.device
        self.blink_params['audio_dim'] = 1024
        self.blink_params['pos_dim'] = 0
        self.blink_params['eye_dim'] = 2

        # 添加归一化参数
        self.max_vals = torch.tensor([90, 90, 90,  1,
            720,  1080]).to(torch.float32).reshape(1, 1, 6)
        self.min_vals = torch.tensor([-90, -90, -90,  0,
            0,  0]).to(torch.float32).reshape(1, 1, 6)

        # 加载模型
        model_p = get_gen_model(self.pose_params)
        model_b = get_gen_model(self.blink_params)

        # 加载预训练权重
        state_dict_p = torch.load(self.pbnet_pose_ckpt, map_location=self.device)
        state_dict_b = torch.load(self.pbnet_blink_ckpt, map_location=self.device)
        model_p.load_state_dict(state_dict_p)
        model_b.load_state_dict(state_dict_b)
        model_p.eval()
        model_b.eval()

        self.model_p = model_p
        self.model_b = model_b

        # 添加视频生成相关默认配置
        current_dir = osp.dirname(osp.abspath(__file__))
        
        # 加载配置文件
        config_path = osp.join(current_dir, 'config', 'DAWN_128.yaml')
        with open(config_path, 'r') as f:
            self.video_config = yaml.safe_load(f)
            
        # 初始化视频生成模型为None，延迟加载
        self.video_model = self._init_video_model(self.video_config['model_config'])

    def switch_conda_env(self, env_name):
        """切换 conda 环境的函数"""
        # 这里需要使用 subprocess 来执行 conda 命令
        subprocess.run(f"conda activate {env_name}", shell=True)
        
    def extract_pose(self):
        """从输入图像中提取人脸姿态和特征点信息。
        
        该函数使用3DDFA-V2模型进行人脸检测和姿态估计。主要步骤包括：
        1. 加载并初始化人脸检测和姿态估计模型
        2. 处理输入图像
        3. 提取人脸姿态和特征点信息
        4. 保存结果到指定路径

        结果文件:
            - init_pose.npy: 包含姿态信息的numpy数组文件
            - init_eye_bbox.npy: 包含眼睛和边界框信息的numpy数组文件
        """
        # 切换到3DDFA环境
        # self.switch_conda_env("3DDFA")
        
        
        
        # 设置ONNX运行环境
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '8'
        
        # 初始化配置
        # config_path = 'configs/mb1_120x120.yml'  # 请确保路径正确
        cfg = yaml.load(open(self.config_path), Loader=yaml.SafeLoader)
        
        # 初始化模型
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
        
        # 读取输入图像
        image = cv2.imread(self.image_path)
        if image.shape[2] == 4:  # 处理RGBA图像
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 人脸检测
        boxes = face_boxes(image)
        if len(boxes) == 0:
            raise ValueError(f'在图像中未检测到人脸: {self.image_path}')
        
        # 获取3DMM参数和ROI框
        param_lst, roi_box_lst = tddfa(image, boxes)
        
        # 重建顶点
        dense_flag = True  # 用于生成密集特征点
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        
        # 获取姿态信息
        pose = get_pose(image, param_lst, ver_lst, show_flag=False, wfp=None, wnp=None)
        
        # 计算眼睛和边界框信息
        lmk = ver_lst[0]
        eye_bbox_result = np.zeros(8)
        bbox = calculate_bbox(image, lmk)
        left_ratio, right_ratio = calculate_eye(lmk)
        
        # 组织结果数据
        eye_bbox_result[0] = left_ratio.sum()
        eye_bbox_result[1] = right_ratio.sum()
        eye_bbox_result[2:] = np.array(bbox)
        
        # 重塑数组
        pose = pose.reshape(1, 7)
        eye_bbox_result = eye_bbox_result.reshape(1, -1)
        
        # 设置保存路径
        eye_bbox_path = os.path.join(self.cache_path, 'init_eye_bbox.npy')
        pose_path = os.path.join(self.cache_path, 'init_pose.npy')
        
        # 保存结果
        np.save(eye_bbox_path, eye_bbox_result)
        np.save(pose_path, pose)

    def process_audio(self):
        """处理音频文件并提取HuBERT特征。

        该方法执行以下步骤：
        1. 将输入音频转换为16kHz采样率
        2. 使用HuBERT模型提取音频特征
        3. 对特征进行插值以匹配视频帧率
        4. 保存处理后的特征

        结果文件:
            - target_audio.npy: 包含插值后的HuBERT特征的numpy数组文件

        Raises:
            RuntimeError: 如果音频处理过程中出现错误
        """
        self.switch_conda_env("DAWN")
        
        try:
            # 创建临时文件用于存储16kHz音频
            with tempfile.NamedTemporaryFile('w', suffix='.wav', dir='./') as temp_wav:
                # 转换音频采样率为16kHz
                self._convert_wav_to_16k(self.audio_path, temp_wav.name)
                
                # 读取16kHz音频
                speech_16k, _ = sf.read(temp_wav.name)
                
                # 计算目标帧数（基于25fps的视频帧率）
                num_frames = int((speech_16k.shape[0] / 16000) * 25)
                
                # 提取HuBERT特征
                hubert_hidden = self._get_hubert_from_16k_speech(speech_16k, device=self.device)
                hubert_hidden = hubert_hidden.detach().numpy()
                
                # 对特征进行线性插值
                interp_func = interp1d(np.arange(hubert_hidden.shape[0]), 
                                    hubert_hidden, 
                                    kind='linear', 
                                    axis=0)
                hubert_feature_interpolated = interp_func(
                    np.linspace(0, hubert_hidden.shape[0] - 1, num_frames)
                ).astype(np.float32)
                
                print(f'Frame count: {num_frames}, HuBERT size: {hubert_hidden.shape[0]}')
                
                # 保存处理后的特征
                np.save(self.audio_emb_path, hubert_feature_interpolated)
                
        except Exception as e:
            raise RuntimeError(f"音频处理失败: {str(e)}")

    def generate_pose_blink(self):
        """生成姿态和眨眼数据。
        
        该函数使用PBnet模型生成驱动的姿态和眨眼数据。主要步骤包括：
        1. 加载预训练的姿态和眨眼模型
        2. 处理输入数据（音频特征、初始姿态、初始眨眼）
        3. 生成驱动数据
        4. 保存结果
        
        结果文件:
            - dri_pose.npy: 生成的姿态数据
            - dri_blink.npy: 生成的眨眼数据
        """
        self.switch_conda_env("DAWN")
        
        # 设置输入路径
        init_pose_path = os.path.join(self.cache_path, 'init_pose.npy')
        init_blink_path = os.path.join(self.cache_path, 'init_eye_bbox.npy')
        
        try:
            # 加载输入数据
            init_pose = torch.from_numpy(np.load(init_pose_path))[:,:self.pose_params['pos_dim']].unsqueeze(0).to(torch.float32)
            init_blink = torch.from_numpy(np.load(init_blink_path))[:,:self.blink_params['eye_dim']].unsqueeze(0).to(torch.float32)
            audio = torch.from_numpy(np.load(self.audio_emb_path)).unsqueeze(0).to(torch.float32)
        except Exception:
            # 3DDFA提取失败时使用默认值
            init_pose = torch.from_numpy(np.array([[0, 0, 0, 4.79e-04, 5.65e+01, 6.49e+01,]]))[:,:self.pose_params['pos_dim']].unsqueeze(0).to(torch.float32)
            init_blink = torch.from_numpy(np.array([[0.3,0.3]]))[:,:self.blink_params['eye_dim']].unsqueeze(0).to(torch.float32)
            audio = torch.from_numpy(np.load(self.audio_emb_path)).unsqueeze(0).to(torch.float32)
        
        # 数据归一化
        init_pose = (init_pose - self.min_vals) / (self.max_vals - self.min_vals)
        
        with torch.no_grad():
            # 生成驱动数据
            gendurations_seg = torch.tensor([audio.shape[1] - 0])
            batch_p = self.model_p.generate(init_pose, audio, gendurations_seg, fact=1)
            batch_b = self.model_b.generate(init_blink, audio, gendurations_seg, fact=1)
            
            # 处理输出数据
            output_p = batch_p['output'].detach().cpu()
            output_b = batch_b['output'].detach().cpu()
            
            output_p = output_p + init_pose
            output_p = inv_transform(output_p, self.min_vals, self.max_vals)
            output_b = output_b + init_blink
            
            # 保存结果
            output_pose_path = os.path.join(self.cache_path, 'dri_pose.npy')
            output_blink_path = os.path.join(self.cache_path, 'dri_blink.npy')
            np.save(output_pose_path, output_p[0])
            np.save(output_blink_path, output_b[0])

    def generate_final_video(self):
        """生成最终视频。
        
        Args:
        Raises:
            RuntimeError: 如果视频生成过程中出现错误
        """
        try:  
            # 准备输出目录
            directory_name = os.path.splitext(os.path.basename(self.image_path))[0]
            video_dir = os.path.join(self.output_path, directory_name, 'video')
            img_dir = os.path.join(self.output_path, directory_name, 'img')
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)
            
            # 准备输入数据
            image = Image.open(self.image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((self.video_config['input_size'], self.video_config['input_size'])),
                transforms.ToTensor()
            ])
            image_tensor = transform(image) * 255
            
            # 加载音频特征和驱动数据
            hubert_npy = np.load(self.audio_emb_path)
            max_frames = min(self.video_config['max_n_frames'], hubert_npy.shape[0])
            ref_hubert = torch.from_numpy(hubert_npy[:max_frames]).to(torch.float32)
            
            drive_poses = torch.from_numpy(np.load(os.path.join(self.cache_path, 'dri_pose.npy'))[:max_frames]).to(torch.float32)
            drive_blink = torch.from_numpy(np.load(os.path.join(self.cache_path, 'dri_blink.npy'))[:max_frames]).to(torch.float32)
            
            try:
                real_poses = torch.from_numpy(np.load(os.path.join(self.cache_path, 'init_pose.npy'))).to(torch.float32)
                real_blink_bbox = torch.from_numpy(np.load(os.path.join(self.cache_path, 'init_eye_bbox.npy'))).to(torch.float32)
            except Exception:
                # 使用默认值
                real_poses = torch.zeros(1, 7)
                real_blink_bbox = torch.tensor([[0.3, 0.3, 64, 64, 192, 192, 256, 256]]).reshape(1, -1).to(torch.float32)
            
            # 准备初始状态
            init_pose = real_poses[0].unsqueeze(0)
            init_blink = real_blink_bbox[0,:2].unsqueeze(0)
            
            # 重排数据维度
            drive_poses = drive_poses.permute(1,0)
            drive_blink = drive_blink.permute(1,0)
            real_blink_bbox = real_blink_bbox.permute(1,0)
            
            # 设置临时文件
            with tempfile.NamedTemporaryFile('w', suffix='.wav') as temp_wav, \
                tempfile.NamedTemporaryFile('w', suffix='.mp4') as temp_video:
                
                # 提取音频片段
                self._extract_audio_segment(self.audio_path, 0, max_frames, 25, temp_wav.name)
                
                # 设置视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    temp_video.name, 
                    fourcc, 
                    25, 
                    (self.video_config['input_size'], self.video_config['input_size'])
                )
                
                # 生成视频帧
                with torch.no_grad():
                    self.video_model.update_num_frames(max_frames)
                    sample_output = self.video_model.sample_one_video(
                        sample_img=image_tensor.unsqueeze(dim=0).cuda()/255.,
                        sample_audio_hubert=ref_hubert.unsqueeze(dim=0).cuda(),
                        sample_pose=drive_poses.unsqueeze(0).cuda(),
                        sample_eye=drive_blink[:2].unsqueeze(0).cuda(),
                        sample_bbox=real_blink_bbox[2:].unsqueeze(0).cuda(),
                        init_pose=init_pose.cuda(),
                        init_eye=init_blink.cuda(),
                        cond_scale=self.video_config['cond_scale']
                    )
                
                # 写入视频帧
                for frame_idx in range(max_frames):
                    frame = self._process_output_frame(
                        sample_output["sample_out_vid"][:, :, frame_idx],
                        mean=self.video_config['mean']
                    )
                    video_writer.write(frame)
                    # 保存每一帧图像
                    frame_name = f"{frame_idx:03d}.png"
                    frame_path = os.path.join(img_dir, frame_name)
                    cv2.imwrite(frame_path, frame)
                video_writer.release()
                
                # 合成最终视频
                output_video_path = os.path.join(video_dir, f"{directory_name}.mp4")
                self._combine_video_audio(temp_wav.name, temp_video.name, output_video_path)
            
        except Exception as e:
            raise RuntimeError(f"视频生成失败: {str(e)}")

    def run(self):
        """执行完整的生成流程"""
        print("1. 提取姿态信息...")
        self.extract_pose()
        
        print("2. 处理音频...")
        self.process_audio()
        
        print("3. 生成姿态和眨眼数据...")
        self.generate_pose_blink()
        
        print("4. 生成最终视频...")
        self.generate_final_video()
    
    @staticmethod
    def _convert_wav_to_16k(input_file, output_file):
        """将音频文件转换为16kHz采样率。

        Args:
            input_file (str): 输入音频文件路径
            output_file (str): 输出音频文件路径
        """
        command = [
            'ffmpeg',
            '-i', input_file,
            '-ar', '16000',
            '-y',  # 添加-y参数以自动覆盖现有文件
            output_file
        ]
        subprocess.run(command)

    @torch.no_grad()
    def _get_hubert_from_16k_speech(self, speech, device="cuda:0"):
        """从16kHz音频中提取HuBERT特征。

        Args:
            speech (numpy.ndarray): 输入音频数据
            device (str): 计算设备，默认为"cuda:0"

        Returns:
            torch.Tensor: HuBERT特征张量

        Notes:
            HuBERT模型使用多层CNN进行处理：
            - 总步长为320（5*2*2*2*2*2）
            - 卷积核大小为400
            - 为避免内存问题，对长音频进行分段处理
        """
        self.hubert_model = self.hubert_model.to(device)
        if speech.ndim == 2:
            speech = speech[:, 0]  # [T, 2] ==> [T,]
            
        input_values_all = self.wav2vec2_processor(
            speech, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_values.to(device)

        # 设置分段处理参数
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
        
        # 分段处理音频
        res_lst = []
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
                
            input_values = input_values_all[:, start_idx: end_idx]
            hidden_states = self.hubert_model(input_values).last_hidden_state
            res_lst.append(hidden_states[0])
        
        # 处理最后一段
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
            
        if input_values.shape[1] >= kernel:
            hidden_states = self.hubert_model(input_values).last_hidden_state
            res_lst.append(hidden_states[0])
        
        # 合并所有特征
        ret = torch.cat(res_lst, dim=0).cpu()
        
        # 确保输出长度正确
        assert abs(ret.shape[0] - expected_T) <= 1
        if ret.shape[0] < expected_T:
            ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
        else:
            ret = ret[:expected_T]
            
        return ret
    

    def _init_video_model(self, model_config):
        """初始化视频生成模型。
        
        Args:
            model_config (dict): 模型配置字典
        
        Returns:
            FlowDiffusion: 初始化好的视频生成模型
        """
        from DM_3.modules.video_flow_diffusion_model_multiGPU_v0_crema_vgg_floss_plus_faceemb_flow_fast_init_cond_test import FlowDiffusion
        
        model = FlowDiffusion(
            is_train=model_config['is_train'],
            sampling_timesteps=self.video_config['sampling_step'],
            ddim_sampling_eta=self.video_config['ddim_sampling_eta'],
            pose_dim=model_config['pose_dim'],
            config_pth=model_config['config_pth'],
            pretrained_pth=model_config['ae_pretrained_pth'],
            win_width=self.video_config['win_width']
        )
        model.cuda()
        
        # 加载预训练权重
        checkpoint = torch.load(model_config['diffusion_pretrained_pth'])
        model.diffusion.load_state_dict(checkpoint['diffusion'])
        model.eval()
        
        return model

    def _process_output_frame(self, frame_batch, mean=(0.0, 0.0, 0.0), index=0):
        """处理模型输出的帧数据。
        
        Args:
            frame_batch (torch.Tensor): 批量帧数据
            mean (tuple): 均值
            index (int): 批次索引
        
        Returns:
            numpy.ndarray: 处理后的BGR格式帧
        """
        frame = frame_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
        frame += np.array(mean)/255.0
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def _extract_audio_segment(self, input_wav, start_frame, num_frames, fps, output_wav):
        """提取音频片段。
        
        Args:
            input_wav (str): 输入音频路径
            start_frame (int): 起始帧
            num_frames (int): 帧数
            fps (int): 帧率
            output_wav (str): 输出音频路径
        """
        
        audio = AudioSegment.from_wav(input_wav)
        frame_duration = 1000 / fps
        start_time = start_frame * frame_duration
        end_time = (start_frame + num_frames) * frame_duration
        audio[start_time:end_time].export(output_wav, format="wav")

    def _combine_video_audio(self, audio_path, video_path, output_path):
            """合并视频和音频。
            
            Args:
                audio_path (str): 音频文件路径
                video_path (str): 视频文件路径
                output_path (str): 输出文件路径
            """
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-i', video_path,
                '-vcodec', 'copy',
                '-ac', '2',
                '-channel_layout', 'stereo',
                '-pix_fmt', 'yuv420p',
                output_path,
                '-shortest'
            ]
            subprocess.run(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, default= 'WRA_MarcoRubio_000.wav', help='输入音频路径')
    parser.add_argument('--image_path', type=str, default= 'real_female_1.jpeg', help='输入图像路径')
    parser.add_argument('--output_path', type=str, default= 'output', help='输出视频路径')
    parser.add_argument('--cache_path', type=str, default='cache/tmp', help='缓存文件路径')
    return parser.parse_args()

def main():
    args = parse_args()
    generator = VideoGenerator(args)
    generator.run()

if __name__ == "__main__":
    main()