import argparse
import os
import sys
sys.path.append('/train20/intern/permanent/lmlin2/ReferenceCode/ACTOR-master')

from src.parser.tools import load_args
from src.parser.base import add_cuda_options, adding_cuda


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpointname")
    parser.add_argument("--dataset", default='crema', help="name of dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--num_frames", default=60, type=int, help="number of frames, if value is bigger than gt nf, load all nf. ")
    parser.add_argument("--niter", default=20, type=int, help="number of iterations")
    parser.add_argument("--num_seq_max", default=3000, type=int, help="number of sequences maximum to load or -1")

    # cuda options
    add_cuda_options(parser)
    
    opt = parser.parse_args()
    newparameters = {key: val for key, val in vars(opt).items() if val is not None}
    
    folder, checkpoint = os.path.split(newparameters["checkpointname"])
    parameters = load_args(os.path.join(folder, "opt.yaml"))
    parameters.update(newparameters)
    adding_cuda(parameters)
    
    if checkpoint.split("_")[0] == 'retraincheckpoint':
        epoch = int(checkpoint.split("_")[2])+int(checkpoint.split("_")[4].split('.')[0])
    else:
        epoch = int(checkpoint.split("_")[1].split('.')[0])
 
    return parameters, folder, checkpoint, epoch, opt.niter


