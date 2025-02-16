import os
import os.path as osp
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.append('.')

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
        
        # Ensure output directories exist
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set intermediate file paths
        self.audio_emb_path = os.path.join(self.cache_path, 'target_audio.npy')

        # Set ONNX runtime environment for 3DDFA
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '8'
        
        # Initialize configuration
        self.config_path = './extract_init_states/configs/mb1_120x120.yml'
        self.cfg = yaml.load(open(self.config_path), Loader=yaml.SafeLoader)
        
        # Initialize models
        self.face_boxes = FaceBoxes_ONNX()
        self.tddfa = TDDFA_ONNX(**self.cfg)

        # HuBERT model configuration
        print("Loading the Wav2Vec2 Processor...")
        self.wav2vec2_processor = AutoProcessor.from_pretrained("/disk2/pfhu/hubert-large-ls960-ft")
        print("Loading the HuBERT Model...")
        self.hubert_model = HubertModel.from_pretrained("/disk2/pfhu/hubert-large-ls960-ft")
        self.hubert_model.eval()
        # PBnet related configuration
        self.pbnet_pose_ckpt = './pretrain_models/pbnet_seperate/pose/checkpoint_40000.pth.tar'
        self.pbnet_blink_ckpt = './pretrain_models/pbnet_seperate/blink/checkpoint_95000.pth.tar'
        self.device = 'cuda:0'
        
        # PBnet model parameters
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

        # Add normalization parameters
        self.max_vals = torch.tensor([90, 90, 90,  1,
            720,  1080]).to(torch.float32).reshape(1, 1, 6)
        self.min_vals = torch.tensor([-90, -90, -90,  0,
            0,  0]).to(torch.float32).reshape(1, 1, 6)

        # Load models
        model_p = get_gen_model(self.pose_params)
        model_b = get_gen_model(self.blink_params)

        # Load pretrained weights
        state_dict_p = torch.load(self.pbnet_pose_ckpt, map_location=self.device)
        state_dict_b = torch.load(self.pbnet_blink_ckpt, map_location=self.device)
        model_p.load_state_dict(state_dict_p)
        model_b.load_state_dict(state_dict_b)
        model_p.eval()
        model_b.eval()

        self.model_p = model_p
        self.model_b = model_b

        # Add default video generation configuration
        current_dir = osp.dirname(osp.abspath(__file__))
        
        # Load configuration file
        config_path = osp.join(current_dir, 'config', 'DAWN_128.yaml')
        with open(config_path, 'r') as f:
            self.video_config = yaml.safe_load(f)
            
        # Initialize video generation model as None for lazy loading
        self.video_model = self._init_video_model(self.video_config['model_config'])

    # def switch_conda_env(self, env_name):
    #     """切换 conda 环境的函数"""
    #     # 这里需要使用 subprocess 来执行 conda 命令
    #     subprocess.run(f"conda activate {env_name}", shell=True)
        
    def extract_pose(self):
        """Extract facial pose and landmark information from input image.
        
        This function uses 3DDFA-V2 model for face detection and pose estimation. Main steps include:
        1. Load and initialize face detection and pose estimation models
        2. Process input image
        3. Extract facial pose and landmark information
        4. Save results to specified paths

        Output files:
            - init_pose.npy: Numpy array file containing pose information
            - init_eye_bbox.npy: Numpy array file containing eye and bounding box information
        """
        
        
        # Set ONNX runtime environment
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '8'
        
        # Initialize configuration
        # config_path = 'configs/mb1_120x120.yml'  # Make sure path is correct
        cfg = yaml.load(open(self.config_path), Loader=yaml.SafeLoader)
        
        # Initialize models
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
        
        # Read input image
        image = cv2.imread(self.image_path)
        if image.shape[2] == 4:  # Handle RGBA images
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Face detection
        boxes = face_boxes(image)
        if len(boxes) == 0:
            raise ValueError(f'No face detected in image: {self.image_path}')
        
        # Get 3DMM parameters and ROI boxes
        param_lst, roi_box_lst = tddfa(image, boxes)
        
        # Reconstruct vertices
        dense_flag = True  # For generating dense landmarks
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        
        # Get pose information
        pose = get_pose(image, param_lst, ver_lst, show_flag=False, wfp=None, wnp=None)
        
        # Calculate eye and bounding box information
        lmk = ver_lst[0]
        eye_bbox_result = np.zeros(8)
        bbox = calculate_bbox(image, lmk)
        left_ratio, right_ratio = calculate_eye(lmk)
        
        # Organize result data
        eye_bbox_result[0] = left_ratio.sum()
        eye_bbox_result[1] = right_ratio.sum()
        eye_bbox_result[2:] = np.array(bbox)
        
        # Reshape arrays
        pose = pose.reshape(1, 7)
        eye_bbox_result = eye_bbox_result.reshape(1, -1)
        
        # Set save paths
        eye_bbox_path = os.path.join(self.cache_path, 'init_eye_bbox.npy')
        pose_path = os.path.join(self.cache_path, 'init_pose.npy')
        
        # Save results
        np.save(eye_bbox_path, eye_bbox_result)
        np.save(pose_path, pose)

    def process_audio(self):
        """Process audio file and extract HuBERT features.
        
        This method performs the following steps:
        1. Convert input audio to 16kHz sampling rate
        2. Extract audio features using HuBERT model 
        3. Interpolate features to match video frame rate
        4. Save processed features

        Output files:
            - target_audio.npy: Numpy array containing interpolated HuBERT features

        Raises:
            RuntimeError: If audio processing fails
        """
        # self.switch_conda_env("DAWN")
        
        try:
            # Create temp file for 16kHz audio
            with tempfile.NamedTemporaryFile('w', suffix='.wav', dir='./') as temp_wav:
                # Convert audio sampling rate to 16kHz
                self._convert_wav_to_16k(self.audio_path, temp_wav.name)
                
                # Read 16kHz audio
                speech_16k, _ = sf.read(temp_wav.name)
                
                # Calculate target frame count (based on 25fps video)
                num_frames = int((speech_16k.shape[0] / 16000) * 25)
                
                # Extract HuBERT features
                hubert_hidden = self._get_hubert_from_16k_speech(speech_16k, device=self.device)
                hubert_hidden = hubert_hidden.detach().numpy()
                
                # Linear interpolation of features
                interp_func = interp1d(np.arange(hubert_hidden.shape[0]), 
                                    hubert_hidden, 
                                    kind='linear', 
                                    axis=0)
                hubert_feature_interpolated = interp_func(
                    np.linspace(0, hubert_hidden.shape[0] - 1, num_frames)
                ).astype(np.float32)
                
                print(f'Frame count: {num_frames}, HuBERT size: {hubert_hidden.shape[0]}')
                
                # Save processed features
                np.save(self.audio_emb_path, hubert_feature_interpolated)
                
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}")

    def generate_pose_blink(self):
        """Generate pose and blink data.
        
        This function uses the PBnet model to generate driving pose and blink data. Main steps include:
        1. Load pretrained pose and blink models
        2. Process input data (audio features, initial pose, initial blink)
        3. Generate driving data
        4. Save results
        
        Output files:
            - dri_pose.npy: Generated pose data
            - dri_blink.npy: Generated blink data
        """
        
        # Set input paths
        init_pose_path = os.path.join(self.cache_path, 'init_pose.npy')
        init_blink_path = os.path.join(self.cache_path, 'init_eye_bbox.npy')
        
        try:
            # Load input data
            init_pose = torch.from_numpy(np.load(init_pose_path))[:,:self.pose_params['pos_dim']].unsqueeze(0).to(torch.float32)
            init_blink = torch.from_numpy(np.load(init_blink_path))[:,:self.blink_params['eye_dim']].unsqueeze(0).to(torch.float32)
            audio = torch.from_numpy(np.load(self.audio_emb_path)).unsqueeze(0).to(torch.float32)
        except Exception:
            # Use default values when 3DDFA extraction fails
            init_pose = torch.from_numpy(np.array([[0, 0, 0, 4.79e-04, 5.65e+01, 6.49e+01,]]))[:,:self.pose_params['pos_dim']].unsqueeze(0).to(torch.float32)
            init_blink = torch.from_numpy(np.array([[0.3,0.3]]))[:,:self.blink_params['eye_dim']].unsqueeze(0).to(torch.float32)
            audio = torch.from_numpy(np.load(self.audio_emb_path)).unsqueeze(0).to(torch.float32)
        
        # normalize
        init_pose = (init_pose - self.min_vals) / (self.max_vals - self.min_vals)
        
        with torch.no_grad():
            # 生成驱动数据
            gendurations_seg = torch.tensor([audio.shape[1] - 0])
            batch_p = self.model_p.generate(init_pose, audio, gendurations_seg, fact=1)
            batch_b = self.model_b.generate(init_blink, audio, gendurations_seg, fact=1)
            
            # process the output
            output_p = batch_p['output'].detach().cpu()
            output_b = batch_b['output'].detach().cpu()
            
            output_p = output_p + init_pose
            output_p = inv_transform(output_p, self.min_vals, self.max_vals)
            output_b = output_b + init_blink
            
            # save results
            output_pose_path = os.path.join(self.cache_path, 'dri_pose.npy')
            output_blink_path = os.path.join(self.cache_path, 'dri_blink.npy')
            np.save(output_pose_path, output_p[0])
            np.save(output_blink_path, output_b[0])

    def generate_final_video(self):
        """Generate the final video.
        
        Args:
        Raises:
            RuntimeError: If an error occurs during video generation
        """
        try:  
            # prepare the output dir
            directory_name = os.path.splitext(os.path.basename(self.image_path))[0]
            video_dir = os.path.join(self.output_path, directory_name, 'video')
            img_dir = os.path.join(self.output_path, directory_name, 'img')
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)
            
            # prepare input
            image = Image.open(self.image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((self.video_config['input_size'], self.video_config['input_size'])),
                transforms.ToTensor()
            ])
            image_tensor = transform(image) * 255
            
            # load the audio emb and condition (pose blink)
            hubert_npy = np.load(self.audio_emb_path)
            max_frames = min(self.video_config['max_n_frames'], hubert_npy.shape[0])
            ref_hubert = torch.from_numpy(hubert_npy[:max_frames]).to(torch.float32)
            
            drive_poses = torch.from_numpy(np.load(os.path.join(self.cache_path, 'dri_pose.npy'))[:max_frames]).to(torch.float32)
            drive_blink = torch.from_numpy(np.load(os.path.join(self.cache_path, 'dri_blink.npy'))[:max_frames]).to(torch.float32)
            
            try:
                real_poses = torch.from_numpy(np.load(os.path.join(self.cache_path, 'init_pose.npy'))).to(torch.float32)
                real_blink_bbox = torch.from_numpy(np.load(os.path.join(self.cache_path, 'init_eye_bbox.npy'))).to(torch.float32)
            except Exception:
                # default value
                real_poses = torch.zeros(1, 7)
                real_blink_bbox = torch.tensor([[0.3, 0.3, 64, 64, 192, 192, 256, 256]]).reshape(1, -1).to(torch.float32)
            
            # prepare init state
            init_pose = real_poses[0].unsqueeze(0)
            init_blink = real_blink_bbox[0,:2].unsqueeze(0)
            
            # process
            drive_poses = drive_poses.permute(1,0)
            drive_blink = drive_blink.permute(1,0)
            real_blink_bbox = real_blink_bbox.permute(1,0)
            
            # temp file
            with tempfile.NamedTemporaryFile('w', suffix='.wav') as temp_wav, \
                tempfile.NamedTemporaryFile('w', suffix='.mp4') as temp_video:
                
                # extract the audio seg
                self._extract_audio_segment(self.audio_path, 0, max_frames, 25, temp_wav.name)
                
                # video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    temp_video.name, 
                    fourcc, 
                    25, 
                    (self.video_config['input_size'], self.video_config['input_size'])
                )
                
                # ddim generation
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
                
                # write the frame
                for frame_idx in range(max_frames):
                    frame = self._process_output_frame(
                        sample_output["sample_out_vid"][:, :, frame_idx],
                        mean=self.video_config['mean']
                    )
                    video_writer.write(frame)
                    # save frames as png
                    frame_name = f"{frame_idx:03d}.png"
                    frame_path = os.path.join(img_dir, frame_name)
                    cv2.imwrite(frame_path, frame)
                video_writer.release()
                
                # save the final video
                output_video_path = os.path.join(video_dir, f"{directory_name}.mp4")
                self._combine_video_audio(temp_wav.name, temp_video.name, output_video_path)
            
        except Exception as e:
            raise RuntimeError(f"! Video generation failed: {str(e)}")

    def run(self):
        """Execute the complete generation pipeline"""
        print("1. Extracting pose information...")
        self.extract_pose()
        
        print("2. Processing audio...")
        self.process_audio()
        
        print("3. Generating pose and blink data...")
        self.generate_pose_blink()
        
        print("4. Generating final video...")
        self.generate_final_video()
    
    @staticmethod
    def _convert_wav_to_16k(input_file, output_file):
        """Convert audio file to 16kHz sampling rate.

        Args:
            input_file (str): Path to input audio file
            output_file (str): Path to output audio file
        """
        command = [
            'ffmpeg',
            '-i', input_file,
            '-ar', '16000',
            '-y',  # Add -y parameter to automatically overwrite existing files
            output_file
        ]
        subprocess.run(command)

    @torch.no_grad()
    def _get_hubert_from_16k_speech(self, speech, device="cuda:0"):
        """Extract HuBERT features from 16kHz audio.

        Args:
            speech (numpy.ndarray): Input audio data
            device (str): Computing device, defaults to "cuda:0"

        Returns:
            torch.Tensor: HuBERT feature tensor

        Notes:
            HuBERT model uses multi-layer CNN for processing:
            - Total stride is 320 (5*2*2*2*2*2)
            - Kernel size is 400
            - Process long audio in segments to avoid memory issues
        """
        self.hubert_model = self.hubert_model.to(device)
        if speech.ndim == 2:
            speech = speech[:, 0]  # [T, 2] ==> [T,]
            
        input_values_all = self.wav2vec2_processor(
            speech, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_values.to(device)

        # Set parameters for segment processing
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
        
        # Process audio in segments
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
        
        # the last seg
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
            
        if input_values.shape[1] >= kernel:
            hidden_states = self.hubert_model(input_values).last_hidden_state
            res_lst.append(hidden_states[0])
        
        # concat the feature
        ret = torch.cat(res_lst, dim=0).cpu()
        
        # check length
        assert abs(ret.shape[0] - expected_T) <= 1
        if ret.shape[0] < expected_T:
            ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
        else:
            ret = ret[:expected_T]
            
        return ret
    

    def _init_video_model(self, model_config):
        """Initialize the video generation model.
        
        Args:
            model_config (dict): Model configuration dictionary
        
        Returns:
            FlowDiffusion: Initialized video generation model
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
        
        # load model
        checkpoint = torch.load(model_config['diffusion_pretrained_pth'])
        model.diffusion.load_state_dict(checkpoint['diffusion'])
        model.eval()
        
        return model

    def _process_output_frame(self, frame_batch, mean=(0.0, 0.0, 0.0), index=0):
        """Process the output frame data from the model.
        
        Args:
            frame_batch (torch.Tensor): Batch of frame data
            mean (tuple): Mean values
            index (int): Batch index
        
        Returns:
            numpy.ndarray: Processed frame in BGR format
        """
        frame = frame_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
        frame += np.array(mean)/255.0
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def _extract_audio_segment(self, input_wav, start_frame, num_frames, fps, output_wav):
        """Extract audio segment.
        
        Args:
            input_wav (str): Input audio path
            start_frame (int): Start frame
            num_frames (int): Number of frames
            fps (int): Frames per second
            output_wav (str): Output audio path
        """
        
        audio = AudioSegment.from_wav(input_wav)
        frame_duration = 1000 / fps
        start_time = start_frame * frame_duration
        end_time = (start_frame + num_frames) * frame_duration
        audio[start_time:end_time].export(output_wav, format="wav")

    def _combine_video_audio(self, audio_path, video_path, output_path):
            """Combine video and audio.
            
            Args:
                audio_path (str): Path to audio file
                video_path (str): Path to video file
                output_path (str): Path to output file
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
    parser.add_argument('--audio_path', type=str, default= 'WRA_MarcoRubio_000.wav', help='Input audio path')
    parser.add_argument('--image_path', type=str, default= 'real_female_1.jpeg', help='Input image path')
    parser.add_argument('--output_path', type=str, default= 'output', help='Output video path')
    parser.add_argument('--cache_path', type=str, default='cache/tmp', help='Cache file path')
    return parser.parse_args()

def main():
    args = parse_args()
    generator = VideoGenerator(args)
    generator.run()

if __name__ == "__main__":
    main()