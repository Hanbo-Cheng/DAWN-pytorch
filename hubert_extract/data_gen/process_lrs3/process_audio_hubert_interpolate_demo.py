import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# adding path of PBnet
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(parent_dir)

from genericpath import exists
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch
from scipy.interpolate import interp1d
import subprocess
import os
from tqdm import tqdm
import tempfile

print("Loading the Wav2Vec2 Processor...")
# wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("./pretrain_models/hubert_ckp")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("./pretrain_models/hubert_ckp", from_tf = True)

def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert

@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    global hubert_model
    print(f"当前显存占用: {torch.cuda.memory_allocated()} 字节")
    print(f"显存缓存占用: {torch.cuda.memory_reserved()} 字节")
    torch.cuda.empty_cache()
    # 强制重置 PyTorch 的 CUDA 分配器
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 可选：手动设置较大的初始缓存大小
    torch.cuda.set_per_process_memory_fraction(0.9)  # 允许使用90%的显存
    # 在加载模型前先检查显存状态
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(torch.cuda.memory_summary())
    
    hubert_model = hubert_model.to(device)
    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

import argparse
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--src_audio_path", default='/train20/intern/permanent/hbcheng2/data/test_speed/target_audio.wav')
    parser.add_argument("--save_path", default='your/path/DAWN-pytorch/ood_data',
                        help="")
    
    return parser.parse_args()



def convert_wav_to_16k(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '16000',
        output_file
    ]
    subprocess.run(command)

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

    args = get_arguments()
    wav_path = args.src_audio_path
    wav_16k_name = wav_path
    npy_name = args.save_path

    output_wav_path = tempfile.NamedTemporaryFile('w', suffix='.wav', dir='./')
    convert_wav_to_16k(wav_path, output_wav_path.name)

    speech_16k, _ = sf.read(output_wav_path.name)
    delete_file(output_wav_path.name)

    # speech_16k, _ = sf.read(wav_path)

    num_frames = int((speech_16k.shape[0] / 16000) * 25)
    hubert_hidden = get_hubert_from_16k_speech(speech_16k, device = 'cuda:0')
    hubert_hidden = hubert_hidden.detach().numpy()
    interp_func = interp1d(np.arange(hubert_hidden.shape[0]), hubert_hidden, kind='linear', axis=0)
    hubert_feature_interpolated = interp_func(np.linspace(0, hubert_hidden.shape[0] - 1, num_frames)).astype(np.float32)
    print(f'fnum{num_frames},hubersize{hubert_hidden.shape[0]}')
    np.save(npy_name, hubert_feature_interpolated)
