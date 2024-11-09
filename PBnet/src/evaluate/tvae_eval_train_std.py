import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.utils.utils import MultiEpochsDataLoader as DataLoader
from src.utils.tensors_hdtf import collate

import os
import numpy as np
import torch.nn.functional as F
# from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model



def evaluate(parameters, dataset, folder, checkpointname, epoch, niter):
    # num_frames = 60

    device = parameters["device"]

    # dummy => update parameters info
    model = get_gen_model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if checkpointname.split("_")[0] == 'retraincheckpoint':
        save_folder = os.path.join(folder, 'fintune_train', checkpointname.split('_')[2]+'_'+checkpointname.split('_')[4].split('.')[0])
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
                    name_list = databatch['videoname']
                    start_list = databatch['start']
                    databatch = {key: val.to(device) for key, val in databatch.items() if key!='videoname' and key!='start'}
                    
                    pose = databatch["x"]
                    audio = databatch["y"]
                    gendurations = databatch["lengths"]
                    # start = databatch["start"]
                    batch = model.forward(databatch)
                    batch = {key: val.to(device) for key, val in batch.items()}
                    

                    for pose_pre, pose_gt, mask, filename, start_num in zip(batch['output'], databatch['x'], databatch['mask'], name_list, start_list):
                        x_ref = pose_gt[0,:].unsqueeze(dim=0).cpu()
                        pose_pre = (pose_pre.cpu()+x_ref - 0.5) * 180
                        gtmasked = (pose_gt[mask].cpu() -0.5 ) * 180
                        outmasked = pose_pre[mask].cpu()
                        pred_path = os.path.join(save_pred_path, filename+'_'+str(start_num))
                        gt_path = os.path.join(save_gt_path, filename+'_'+str(start_num)+'_gt')
                        # np.save(pred_path, pose_pre.cpu())
                        # np.save(gt_path, pose_gt.cpu())
                        np.savetxt(pred_path, outmasked)
                        np.savetxt(gt_path, gtmasked)
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
