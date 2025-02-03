import torch
from tqdm import tqdm
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(parent_dir)

from src.utils.fixseed import fixseed
from src.parser.tools import load_args
import os
import numpy as np
import torch.nn.functional as F
# from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
import argparse

max_vals = torch.tensor([90, 90, 90,  1,
            720,  1080]).to(torch.float32).reshape(1, 1, 6)
min_vals = torch.tensor([-90, -90, -90,  0,
            0,  0]).to(torch.float32).reshape(1, 1, 6)

def inv_transform(x, min_val, max_val):
    out = x * (max_val - min_val) + min_val
    return out

def save_images_as_npy(input_data, output_file):
    # save_npy = np.zeros(input_data.shape[0], 7)
    # save_npy[:,:, :-1] = input_data
    # save_npy[:, -1] = ref[:,:, -1]
    # images_array = np.array(images)
    np.save(output_file, input_data)




# def transform(x, min_val, max_val):
#     out = x * (max_val - min_val) + min_val
#     return out

def evaluate(parameters_pose, parameters_blink, audio_path, init_pose_path, init_blink_path, checkpoint_p_path, checkpoint_b_path, output_path):
    # num_frames = 60
    # min_val = dataset.min_vals
    # max_val = dataset.max_vals
    device = "cuda:0"
    pose_dim = parameters_pose['pos_dim']
    eye_dim = parameters_blink['eye_dim']
    # dummy => update parameters info
    model_p = get_gen_model(parameters_pose)
    model_b = get_gen_model(parameters_blink)

    print("Restore weights..")
    # checkpointpath = os.path.join(folder, checkpointname)
    # model_p_ckpt = model_p.state_dict()
    # model_b_ckpt = model_b.state_dict()
    state_dict_p = torch.load(checkpoint_p_path, map_location=device)
    state_dict_b = torch.load(checkpoint_b_path, map_location=device)
    # for name, _ in model_ckpt.items():
    #     if  model_ckpt[name].shape == state_dict[name].shape:
    #         model_ckpt[name].copy_(state_dict[name])
    #     model.load_state_dict(model_ckpt)
    model_p.load_state_dict(state_dict_p)
    model_b.load_state_dict(state_dict_b)
    model_p.eval()
    model_b.eval()

    os.makedirs(output_path, exist_ok=True)

    try:
        init_pose = torch.from_numpy(np.load(init_pose_path))[:,:pose_dim].unsqueeze(0).to(torch.float32)
        init_blink = torch.from_numpy(np.load(init_blink_path))[:,:eye_dim].unsqueeze(0).to(torch.float32)
        audio = torch.from_numpy(np.load(audio_path)).unsqueeze(0).to(torch.float32)
    except Exception:
        # the 3ddfa fail to extract valid pose, using typical value instead
        init_pose = torch.from_numpy(np.array([[0, 0, 0, 4.79e-04, 5.65e+01, 6.49e+01,]]))[:,:pose_dim].unsqueeze(0).to(torch.float32)
        init_blink = torch.from_numpy(np.array([[0.3,0.3]]))[:,:eye_dim].unsqueeze(0).to(torch.float32)
        audio = torch.from_numpy(np.load(audio_path)).unsqueeze(0).to(torch.float32)

    init_pose = (init_pose - min_vals)/ (max_vals - min_vals)
    fixseed(1234)
        

    with torch.no_grad():

        # batch = {key: val.to(device) for key, val in databatch.items()}


        # step 1: seg
        pose_seg = init_pose
        blink_seg = init_blink
        audio_seg = audio
        # gendurations_seg = gendurations[:, start:end]
        gendurations_seg = torch.tensor([audio.shape[1] - 0])
        # step 2: predict
        batch_p = model_p.generate(pose_seg, audio_seg, gendurations_seg, fact = 1)
        batch_b = model_b.generate(blink_seg, audio_seg, gendurations_seg, fact = 1)
        # step 3: merge

        output_p = batch_p['output'].detach().cpu()
        output_b = batch_b['output'].detach().cpu()

        output_p = output_p + pose_seg
        output_p = inv_transform(output_p, min_vals, max_vals)
        output_b = output_b + blink_seg

        output_pose_path = os.path.join(output_path, 'dri_pose.npy')
        output_blink_path = os.path.join(output_path, 'dri_blink.npy')

        np.save(output_pose_path , output_p[0])
        np.save(output_blink_path, output_b[0])

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PBnet")
    parser.add_argument("--audio_path", default='/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wav_hubert_interpolate/RD_Radio54_000.npy')
    parser.add_argument("--ckpt_pose", default='your_path/pretrain_models/pbnet_seperate/pose/checkpoint_40000.pth.tar',
                        help="ckpt of PoseNet")
    parser.add_argument("--ckpt_blink", default='your_path/pretrain_models/pbnet_seperate/blink/checkpoint_95000.pth.tar',
                        help="ckpt of BlinkNet")

    parser.add_argument("--init_pose_blink", default='your/path/DAWN-pytorch/ood_data/ood_test_material/cache_2',
                        help="dir of init pose/blink")
    
    parser.add_argument("--output", default='/train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/demo_output',
                        help="output_dir")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    audio_path = args.audio_path
    ckpt_pose = args.ckpt_pose
    ckpt_blink = args.ckpt_blink
    output_dir = args.output
    init_blink = os.path.join(args.init_pose_blink, 'init_eye_bbox.npy') # init_eye_bbox.npy
    init_pose = os.path.join(args.init_pose_blink, 'init_pose.npy')

    folder_p, _ = os.path.split(ckpt_pose)
    parameters_p = load_args(os.path.join(folder_p, "opt.yaml"))
    parameters_p['device'] = 'cuda:0'
    parameters_p["audio_dim"] = 1024
    parameters_p["pos_dim"] = 6
    parameters_p["eye_dim"] = 0

    folder_b, _ = os.path.split(ckpt_blink)
    parameters_b = load_args(os.path.join(folder_b, "opt.yaml"))
    parameters_b['device'] = 'cuda:0'
    parameters_b["audio_dim"] = 1024
    parameters_b["pos_dim"] = 0
    parameters_b["eye_dim"] = 2

    evaluate(parameters_pose = parameters_p,
            parameters_blink = parameters_b,
            audio_path = audio_path,
            init_pose_path = init_pose,
            init_blink_path = init_blink,
            checkpoint_p_path = ckpt_pose,
            checkpoint_b_path = ckpt_blink,
            output_path = output_dir)


