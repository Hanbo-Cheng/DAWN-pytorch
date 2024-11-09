import torch
from tqdm import tqdm
import os
import sys
# adding path of PBnet
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(parent_dir)

from src.utils.fixseed import fixseed
from src.parser.tools import load_args
import numpy as np
import torch.nn.functional as F
# from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
import argparse

max_vals = torch.tensor([90, 90, 90,  1,
            720,  1080, 1, 1]).to(torch.float32).reshape(1, 1, 8)
min_vals = torch.tensor([-90, -90, -90,  0,
            0,  0, 0, 0]).to(torch.float32).reshape(1, 1, 8)

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

def evaluate(parameters, audio_path, init_pose_path, init_blink_path, checkpoint_path, output_path):
    # num_frames = 60
    # min_val = dataset.min_vals
    # max_val = dataset.max_vals
    device = parameters["device"]
    pose_dim = parameters['pos_dim']
    eye_dim = parameters['eye_dim']
    # dummy => update parameters info
    model = get_gen_model(parameters)

    print("Restore weights..")
    # checkpointpath = os.path.join(folder, checkpointname)
    # model_p_ckpt = model_p.state_dict()
    # model_b_ckpt = model_b.state_dict()
    state_dict_p = torch.load(checkpoint_path, map_location=device)
    # for name, _ in model_ckpt.items():
    #     if  model_ckpt[name].shape == state_dict[name].shape:
    #         model_ckpt[name].copy_(state_dict[name])
    #     model.load_state_dict(model_ckpt)
    model.load_state_dict(state_dict_p)

    model.eval()


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

    pose_seg = init_pose
    blink_seg = init_blink
    init_pose = torch.concat([pose_seg, blink_seg], dim = -1)

    init_pose = (init_pose - min_vals)/ (max_vals - min_vals)
    fixseed(1234)
        

    with torch.no_grad():

        # batch = {key: val.to(device) for key, val in databatch.items()}


        # step 1: seg
        
        audio_seg = audio
        # gendurations_seg = gendurations[:, start:end]
        gendurations_seg = torch.tensor([audio.shape[1] - 0])
        # step 2: predict
        batch = model.generate(init_pose, audio_seg, gendurations_seg, fact = 1)
        # step 3: merge

        output = batch['output'].detach().cpu()

        output = output + init_pose
        output = inv_transform(output, min_vals, max_vals)

        output_pose_path = os.path.join(output_path, 'dri_pose.npy')
        output_blink_path = os.path.join(output_path, 'dri_blink.npy')

        np.save(output_pose_path , output[0,:,:pose_dim])
        np.save(output_blink_path, output[0,:,pose_dim:])

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PBnet")
    parser.add_argument("--audio_path", default='/train20/intern/permanent/hbcheng2/data/HDTF/hdtf_wav_hubert_interpolate/RD_Radio54_000.npy')
    parser.add_argument("--ckpt", default='../pretrain_models/pbnet_both/checkpoint_100000.pth.tar',
                        help="ckpt of PoseNet")

    parser.add_argument("--init_pose_blink", default='your/path/DAWN-pytorch/ood_data/ood_test_material/cache_2',
                        help="dir of init pose/blink")
    
    parser.add_argument("--output", default='/train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/demo_output',
                        help="output_dir")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    audio_path = args.audio_path
    ckpt_pose = args.ckpt
    output_dir = args.output
    init_blink = os.path.join(args.init_pose_blink, 'init_eye_bbox.npy') # init_eye_bbox.npy
    init_pose = os.path.join(args.init_pose_blink, 'init_pose.npy')

    folder_p, _ = os.path.split(ckpt_pose)
    parameters = load_args(os.path.join(folder_p, "opt.yaml"))
    parameters['device'] = 'cuda:0'
    parameters["audio_dim"] = 1024
    parameters["pos_dim"] = 6
    parameters["eye_dim"] = 2


    evaluate(parameters = parameters,
            audio_path = audio_path,
            init_pose_path = init_pose,
            init_blink_path = init_blink,
            checkpoint_path = ckpt_pose,
            output_path = output_dir)


