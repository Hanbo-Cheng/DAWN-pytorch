import torch
import torch.nn as nn

from ..tools.losses import get_loss_function
import torch.nn.functional as F
# from ..rotation2xyz import Rotation2xyz
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from src.models.architectures.tools.resnet import resnet34

class MyResNet34(nn.Module):
    def __init__(self,embedding_dim,input_channel = 3):
        super(MyResNet34, self).__init__()
        self.resnet = resnet34(norm_layer = BatchNorm2d,num_classes=embedding_dim,input_channel = input_channel)
    def forward(self, x):
        return self.resnet(x)


class LSTM(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, **kwargs):
        super(LSTM,self).__init__()

        self.em_audio = MyResNet34(256, 1)
        self.em_init_pose = nn.Linear(3,256)

        self.lstm = nn.LSTM(512,256,num_layers=2,bias=True,batch_first=True)
        self.output = nn.Linear(256,3)

        self.lambdas = lambdas
        self.losses = list(self.lambdas) + ["mixed"]
        self.device = device

    def compute_loss(self, batch):
        mixed_loss = 0
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss*lam
            losses[ltype] = loss.item()
        
        # D_loss, G_loss = self.calculate_GAN_loss(batch)
        # mixed_loss += G_loss * 0.7
        # losses['GAN_D'] = D_loss
        # losses['GAN_G'] = D_loss
        losses["mixed"] = mixed_loss.item()
        return mixed_loss, losses
        
    def forward(self,batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs = x.shape[0]
        x_ref = x[:,0,:].unsqueeze(dim=1) # The pose information of the first frame(refrence img)
        x = x-x_ref.repeat(1,x.size(1),1) # bs, nf, 6  Obtain the difference from the first frame
        batch['x_delta'] = x
        ref_pose = self.em_init_pose(batch["x"][:,0,:])
        result = []
        bs,seqlen,_,_ = batch["y"].shape
        zero_state = torch.zeros((2,bs,256),requires_grad=True).to(ref_pose.device)
        cur_state = (zero_state,zero_state)
        audio = batch["y"].reshape(-1, 1, 4, 41)
        audio_em = self.em_audio(audio).reshape(bs, seqlen, 256)
        for i in range(seqlen):

            ref_pose,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],ref_pose.unsqueeze(1)),dim=2),cur_state)
            ref_pose = ref_pose.reshape(-1, 256)
            result.append(self.output(ref_pose).unsqueeze(1))
        res = torch.cat(result,dim=1)
        batch['output'] = res
        return batch

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate(self, pose, audio, durations,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1):
        
        x = pose.to(self.device)
        y = audio.to(self.device) 

        if len(durations.shape) == 1:
            lengths = durations.to(self.device)
        else:
            lengths = durations.to(self.device).reshape(y.shape)
        
        mask = self.lengths_to_mask(lengths)
        batch = {"x": x, "y": y, "mask": mask, "lengths": lengths}
        batch = self.forward(batch)
        
        return batch
