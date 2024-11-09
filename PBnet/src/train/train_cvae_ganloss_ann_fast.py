import sys
sys.path.append('your_path/PBnet')

import os
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.utils import MultiEpochsDataLoader as DataLoader
from src.train.trainer_gan_ann import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.datasets.datasets_crema_pos import CREMA
from src.datasets.datasets_hdtf_pos_chunk_norm_2 import HDTF

import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "kl", "rcw", "ssim"]  # not used: "hp", "mmd", "vel", "velxyz"

MODELTYPES = ["cvae"]  # not used: "cae"
ARCHINAMES = ["fc", "gru", "transformer", "transgru", "grutrans", "autotrans"]

class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm='batch', leaky=True):
        super(ConvNormRelu, self).__init__()
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
        if norm == 'batch':
            layers.append(nn.BatchNorm1d(out_channels))
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    

class D_patchgan(nn.Module):
    def __init__(self, n_downsampling=2, norm='batch'):
        super(D_patchgan, self).__init__()
        ndf = 64
        self.conv1 = nn.Conv1d(6, ndf, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        layers = []
        for n in range(0, n_downsampling):
            nf_mult = min(2**n, 8)
            layers.append(ConvNormRelu(ndf * nf_mult, ndf * nf_mult * 2, kernel_size=4, stride=2, padding=1, norm=norm))

        nf_mult = min(2**n_downsampling, 8)
        layers.append(ConvNormRelu(ndf * nf_mult, ndf * nf_mult, kernel_size=4, stride=1, padding=1, norm=norm))

        layers.append(nn.Conv1d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.model(out)
        return out

    def calculate_GAN_loss(self, batch):
            x = batch["x"] #bs, nf, 6
            x_ref = x[:,0,:].unsqueeze(dim=1) #bs, 1, 64
            output = batch["output"]+x_ref #bs, nf, 6

            real_pose_score = self.forward(x.permute(0,2,1))
            fake_pose_score = self.forward(output.permute(0,2,1))

            D_loss = F.binary_cross_entropy_with_logits(real_pose_score, torch.ones_like(real_pose_score)) + F.binary_cross_entropy_with_logits(fake_pose_score, torch.zeros_like(fake_pose_score))
            G_loss = F.binary_cross_entropy_with_logits(fake_pose_score, torch.ones_like(fake_pose_score))

            return D_loss.mean(), G_loss.mean()


def get_model(parameters):
    modeltype = parameters["modeltype"]
    archiname = parameters["archiname"]

    archi_module = importlib.import_module(f'.architectures.{archiname}', package="src.models")
    Encoder = archi_module.__getattribute__(f"Encoder_{archiname.upper()}")
    Decoder = archi_module.__getattribute__(f"Decoder_{archiname.upper()}")

    model_module = importlib.import_module(f'.modeltype.{modeltype}', package="src.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")

    encoder = Encoder(**parameters)
    decoder = Decoder(**parameters)
    
    # parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    return Model(encoder, decoder, **parameters).to(parameters["device"])

def do_epochs(model, model_d, dataset, parameters, optimizer_g, optimizer_d, scheduler_g, scheduler_d, writer):
    # train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
    #                             shuffle=True, num_workers=8, pin_memory=True)
    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=8, collate_fn=collate, pin_memory = True)

    logpath = os.path.join(parameters["folder"], "training.log")
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
            dict_loss = train(model, model_d, optimizer_g, optimizer_d, train_iterator, model.device, epoch)

            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)
            scheduler_g.step()
            scheduler_d.step()
            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                torch.save(model.state_dict(), checkpoint_path)

            writer.flush()


if __name__ == '__main__':

    # parse options
    parameters = parser()
    
    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    os.environ["CUDA_VISIBLE_DEVICES"] = parameters["gpu"]
    dataset_name = parameters["dataset"]
    if dataset_name == 'crema':
        # data path
        data_dir = "/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/images"
        # model and dataset
        dataset = CREMA(data_dir=data_dir,
                        max_num_frames=parameters["num_frames"],
                        mode = 'train')
        dataset.update_parameters(parameters)
    elif dataset_name == 'hdtf':
        data_dir = "/yrfs2/cv2/pcxia/audiovisual/hdtf/images_25hz"
        dataset = HDTF(data_dir=data_dir,
                        max_num_frames=parameters["num_frames"],
                        mode = 'train')
        dataset.update_parameters(parameters)
    else:
        dataset = None
        print('Dataset can not be found!!')

    

    model = get_model(parameters)
    model_d = D_patchgan().to(parameters["device"])
    # optimizer
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=parameters["lr"])
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=parameters["num_epochs"], eta_min=2e-5)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=parameters["num_epochs"], eta_min=2e-5)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    do_epochs(model, model_d, dataset, parameters, optimizer_g, optimizer_d, scheduler_g, scheduler_d, writer)

    writer.close()
