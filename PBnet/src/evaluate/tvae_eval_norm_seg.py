import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.utils.utils import MultiEpochsDataLoader as DataLoader
from src.utils.tensors_hdtf import collate, collate_old

import os
import numpy as np
import torch.nn.functional as F
# from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model

INF_LENGTH = 600

def save_images_as_npy(input_data, output_file):
    # save_npy = np.zeros(input_data.shape[0], 7)
    # save_npy[:,:, :-1] = input_data
    # save_npy[:, -1] = ref[:,:, -1]
    # images_array = np.array(images)
    np.save(output_file, input_data)


def save_as_chunk(dir, data): 
    if not os.path.exists(dir):
        os.makedirs(dir)
    chunks = [data[i:min(i + 25, data.shape[0])] for i in range(0, data.shape[0], 25)]

    for i, chunk in enumerate(chunks):
        output_file = os.path.join(dir, f'chunk_%04d.npy' % (i))
        # chunk = np.stack(chunk, axis = 0)
        save_images_as_npy(chunk, output_file)



def transform(x, min_val, max_val):
    out = x * (max_val - min_val) + min_val
    return out

def evaluate(parameters, dataset, folder, checkpointname, epoch, niter):
    # num_frames = 60
    min_val = dataset.min_vals
    max_val = dataset.max_vals
    device = parameters["device"]

    # dummy => update parameters info
    model = get_gen_model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    model_ckpt = model.state_dict()
    state_dict = torch.load(checkpointpath, map_location=device)
    for name, _ in model_ckpt.items():
        if  model_ckpt[name].shape == state_dict[name].shape:
            model_ckpt[name].copy_(state_dict[name])
        model.load_state_dict(model_ckpt)
    # model.load_state_dict(state_dict)
    model.eval()

    if checkpointname.split("_")[0] == 'retraincheckpoint':
        save_folder = os.path.join(folder, 'fintune', checkpointname.split('_')[2]+'_'+checkpointname.split('_')[4].split('.')[0])
    else:
        save_folder = os.path.join(folder, 'nofinetune', checkpointname.split('_')[1].split('.')[0])
    os.makedirs(save_folder, exist_ok=True)

    allseeds = list(range(niter))

    try:
        for index, seed in enumerate(allseeds):
            print(f"Evaluation number: {index+1}/{niter}")
            fixseed(seed)
            save_pred_path = os.path.join(save_folder, 'eval_pred', str(seed))
            save_gt_path = os.path.join(save_folder, 'eval_gt', str(seed))
            os.makedirs(save_pred_path, exist_ok=True)
            os.makedirs(save_gt_path, exist_ok=True)
            

            dataiterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=False, num_workers=8, collate_fn=collate_old)

            with torch.no_grad():
                for databatch in tqdm(dataiterator, desc=f"Construct dataloader: generating.."):
                    # batch = {key: val.to(device) for key, val in databatch.items()}
                    pose = databatch["x"][:,:,:-1]   # b, len, c
                    ref = databatch['x'][:,:, -1]
                    audio = databatch["y"]  # b, len, c
                    gendurations = databatch["lengths"]
                    # start = databatch["start"]

                    output = None
                    for i in range(0, pose.shape[1], INF_LENGTH):
                        # step 1: seg
                        start = i
                        end = min(pose.shape[1], i + INF_LENGTH)
                        pose_seg = pose[:, start:end]
                        audio_seg = audio[:, start:end]
                        gendurations_seg = torch.tensor([end - start])
                        # step 2: predict
                        batch = model.generate(pose_seg, audio_seg, gendurations_seg, fact = 1)
                        # step 3: merge
                        if output == None:
                            output = batch['output'].detach().cpu()
                        else:
                            output = torch.concat([output, batch['output'].detach().cpu()], dim= 1)
                        



                    
                    # batch = model.generate(pose, audio, gendurations, fact = 1)
                    # batch = {key: val.to(device) for key, val in batch.items()}
                    

                    for pose_pre, pose_gt, mask, filename, start_num in zip(output, databatch['x'], databatch['mask'], databatch['videoname'], databatch['start']):
                        
                        pose_pre = pose_pre.cpu()
                        padding_vec = torch.zeros(pose_pre.shape[0], 1)
                        pose_pre = torch.concat([pose_pre, padding_vec], dim = -1)
                        for i in range(0, pose_gt.shape[0], INF_LENGTH):
                            start = i
                            end = min(pose_gt.shape[0], i + INF_LENGTH)
                            x_ref = pose_gt[i,:].unsqueeze(dim=0)
                            pose_pre[start:end] = pose_pre[start:end]+x_ref
                        gtmasked = pose_gt[mask].cpu()
                        outmasked = pose_pre[mask].cpu()
                        gtmasked = transform(gtmasked, min_val, max_val)
                        outmasked = transform(outmasked, min_val, max_val)
                        pred_dir = os.path.join(save_pred_path, filename)
                        save_as_chunk(pred_dir, outmasked)
                        # np.save(pred_path, pose_pre.cpu())
                        # np.save(gt_path, pose_gt.cpu())
                        # np.savetxt(pred_path, outmasked)
                        # np.savetxt(gt_path, gtmasked)
                        loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
                        print(loss)
                        


    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)


    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    # save_metrics(evalpath, metrics)
