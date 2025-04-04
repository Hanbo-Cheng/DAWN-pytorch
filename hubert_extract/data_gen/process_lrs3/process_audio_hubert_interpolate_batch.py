from genericpath import exists
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
import fairseq
import decord
from scipy.interpolate import interp1d


print("Loading the Wav2Vec2 Processor...")
# wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("/train20/intern/permanent/hbcheng2/lmlin2/Flow/GeneFace-main/checkpoints/hubert_ckp")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("/train20/intern/permanent/hbcheng2/lmlin2/Flow/GeneFace-main/checkpoints/hubert_ckp", from_tf = True)

def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert

@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:3"):
    global hubert_model
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

# decord.bridge.set_bridge('torch')
from tqdm import tqdm
if __name__ == '__main__':
    ## Process Single Long Audio for NeRF dataset
    # person_id = 'May'
    import os

    # demo test

    # wav_16k_name = f"/train20/intern/permanent/lmlin2/data/audio_lesson_01-j-w.wav"
    # npy_name = 'demo_test-j-w'
    # demo_npy_name = f"/train20/intern/permanent/lmlin2/data/{npy_name}.npy"
    # speech_16k, _ = sf.read(wav_16k_name)
    # hubert_hidden = get_hubert_from_16k_speech(speech_16k)
    # np.save(demo_npy_name, hubert_hidden.detach().numpy())

    # hdtf dataset
    # image_path = '/train20/intern/permanent/lmlin2/data/hdtf_image_50hz'
    # video_path = '/train20/intern/permanent/hbcheng2/data/HDTF/video_25hz'
    # wav_path = '/yrfs2/cv2/pcxia/audiovisual/hdtf/image_audio'
    # save_path = '/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wav_hubert'
    # interpolate_path = "/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wav_hubert_interpolate"

    # if not os.path.exists(interpolate_path):
    #     os.makedirs(interpolate_path)

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # for wavfile in tqdm(os.listdir(wav_path)):
    #     video = os.path.join(video_path, wavfile[0:-4] + '.mp4')
    #     vr = decord.VideoReader(video)
    #     num_frames = len(vr)
    #     wav_16k_name = os.path.join(wav_path, wavfile)
    #     # wav_16k_name = f"/yrfs2/cv2/pcxia/audiovisual/hdtf/image_audio/RD_Radio1_000.wav"           #(3749, 1024)
    #     # wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"          
    #     # wav_16k_name = f"/train20/intern/permanent/lmlin2/Flow/GeneFace-main/data/raw/val_wavs/zozo.wav"  # 543 1024
    #     # wav_16k_name = f"/train20/intern/permanent/lmlin2/data/audio_lesson_01.wav"
    #     npy_name = wavfile[0:-4]
    #     hubert_npy_name = f"{save_path}/{npy_name}.npy"
    #     hubert_npy_name_interpolate = f"{interpolate_path}/{npy_name}.npy"
    #     if os.path.exists(hubert_npy_name) and os.path.exists(hubert_npy_name_interpolate):
    #         continue
    #     speech_16k, _ = sf.read(wav_16k_name)
    #     hubert_hidden = get_hubert_from_16k_speech(speech_16k)
    #     print(f'fnum{num_frames},hubersize{hubert_hidden.shape[0]}')

    #     hubert_hidden = hubert_hidden.detach().numpy()
    #     interp_func = interp1d(np.arange(hubert_hidden.shape[0]), hubert_hidden, kind='linear', axis=0)
    #     hubert_feature_interpolated = interp_func(np.linspace(0, hubert_hidden.shape[0] - 1, num_frames)).astype(np.float32)
    #     # torch.nn.functional.interpolate(hubert_hidden.unsqueeze(0).permute(0,2,1).cuda(), size=num_frames, mode='linear', align_corners=False).squeeze(0).permute(1, 0).cpu()
    #     np.save(hubert_npy_name, hubert_hidden)
    #     np.save(hubert_npy_name_interpolate, hubert_feature_interpolated)


    # crema dataset
    # image_path = '/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/images'
    # wav_path = '/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/audio'
    # save_path = '/train20/intern/permanent/hbcheng2/data/crema/hubert_25hz'
    wav_path = '/train20/intern/permanent/hbcheng2/data/ood_video/audio_clip_2'
    save_path = '/train20/intern/permanent/hbcheng2/data/ood_video/audio_clip_hubert'
    # for id_name in tqdm(os.listdir(wav_path)):
    for wavfile in os.listdir(wav_path):
            # frame_dir = os.path.join(image_path, id_name, wavfile[0:-4])
            # if not exists(frame_dir):
            #     print(f'{frame_dir} does not exist!')
            #     continue
            # frames= os.listdir(frame_dir)
            # frames.sort()
            # num_frames = len(frames)
            wav_16k_name = os.path.join(wav_path, wavfile)
            npy_name = wavfile[0:-4]
            save_dir = os.path.join(save_path)
            hubert_npy_name = os.path.join(save_dir,npy_name+'.npy') 
            # if exists(hubert_npy_name):
            #     print(f'{hubert_npy_name} exists!')
            #     continue
            if not exists(save_dir):
                os.makedirs(save_dir)
            
            speech_16k, _ = sf.read(wav_16k_name)
            num_frames = int((speech_16k.shape[0] / 16000) * 25)
            hubert_hidden = get_hubert_from_16k_speech(speech_16k)
            hubert_hidden = hubert_hidden.detach().numpy()
            interp_func = interp1d(np.arange(hubert_hidden.shape[0]), hubert_hidden, kind='linear', axis=0)
            hubert_feature_interpolated = interp_func(np.linspace(0, hubert_hidden.shape[0] - 1, num_frames)).astype(np.float32)
            # print(f'fnum{num_frames},hubersize{hubert_hidden.shape[0]}')
            np.save(hubert_npy_name, hubert_feature_interpolated)

    # ## Process short audio clips for LRS3 dataset
    # import glob, os, tqdm
    # lrs3_dir = '/home/yezhenhui/datasets/raw/lrs3_raw/'
    # wav_16k_names = glob.glob(os.path.join(lrs3_dir, '*/*.wav'))
    # for wav_16k_name in tqdm.tqdm(wav_16k_names, total=len(wav_16k_names)):
    #     spk_id = wav_16k_name.split("/")[-2]
    #     clip_id = wav_16k_name.split("/")[-1][:-4]
    #     out_name = os.path.join(lrs3_dir, spk_id, clip_id+'_hubert.npy')
    #     if os.path.exists(out_name):
    #         continue
    #     speech_16k, _ = sf.read(wav_16k_name)
    #     hubert_hidden = get_hubert_from_16k_speech(speech_16k)
    #     np.save(out_name, hubert_hidden.detach().numpy())