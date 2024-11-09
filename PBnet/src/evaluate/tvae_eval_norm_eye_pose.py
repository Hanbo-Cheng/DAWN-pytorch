import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.utils.utils import MultiEpochsDataLoader as DataLoader
from src.utils.tensors_eye_eval import collate

import os
import numpy as np
import torch.nn.functional as F
import time
# from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model


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
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
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
                                shuffle=False, num_workers=8, collate_fn=collate)

            with torch.no_grad():
                for databatch in tqdm(dataiterator, desc=f"Construct dataloader: generating.."):
                    # batch = {key: val.to(device) for key, val in databatch.items()}
                    pose_eye = databatch["x"]
                    audio = databatch["y"]
                    gendurations = databatch["lengths"]
                    # start = databatch["start"]
                    start_time = time.time()
                    batch = model.generate(pose_eye, audio, gendurations, fact = 1)
                    end_time = time.time()
                    print(f'generate audio time {end_time- start_time}')
                    start_time = end_time
                    # exit()
                    batch = {key: val.to(device) for key, val in batch.items()}
                    

                    for pose_eye_pre, pose_eye_gt, mask, filename, start_num in zip(batch['output'], databatch['x'], databatch['mask'], databatch['videoname'], databatch['start']):
                        x_ref = pose_eye_gt[0,:].unsqueeze(dim=0)
                        pose_eye_pre = pose_eye_pre.cpu()+x_ref
                        gtmasked = pose_eye_gt[mask].cpu()
                        outmasked = pose_eye_pre[mask].cpu()
                        gtmasked[:,:-2] = transform(gtmasked[:,:-2], min_val, max_val)
                        outmasked[:,:-2] = transform(outmasked[:,:-2], min_val, max_val)
                        pred_path = os.path.join(save_pred_path, filename+'_'+str(start_num))
                        gt_path = os.path.join(save_gt_path, filename+'_'+str(start_num)+'_gt')
                        # np.save(pred_path, pose_pre.cpu())
                        # np.save(gt_path, pose_gt.cpu())
                        np.savetxt(pred_path, outmasked)
                        np.savetxt(gt_path, gtmasked)
                        loss = F.mse_loss(gtmasked[:,:3], outmasked[:,:3], reduction='mean')
                        print(loss)
                        


    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)


    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    # save_metrics(evalpath, metrics)
