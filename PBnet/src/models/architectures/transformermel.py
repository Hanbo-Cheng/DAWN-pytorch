import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.architectures.tools.util import MyResNet34


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

        
# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

class Encoder_TRANSFORMERMEL(nn.Module):
    def __init__(self, modeltype, num_frames, audio_dim=1024, pos_dim=6, pose_latent_dim=64,
                 audio_latent_dim=256, ff_size=128, num_layers=2, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        
        self.modeltype = modeltype
        self.pos_dim = pos_dim
        self.num_frames = num_frames
        self.audio_dim = audio_dim
        
        self.pose_latent_dim = pose_latent_dim
        self.audio_latent_dim = audio_latent_dim
        self.latent_dim = self.audio_latent_dim + self.pose_latent_dim*2
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        
        # if self.ablation == "average_encoder":
        #     self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
        #     self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        # else:
        #     self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        #     self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        # # there's no class of our dataset CREMA/HDTF, so noly  dont need to use nn.parameter
        self.mu_layer = nn.Linear(self.latent_dim, self.audio_latent_dim)
        self.sigma_layer = nn.Linear(self.latent_dim, self.audio_latent_dim)
        
        self.poseEmbedding = nn.Linear(self.pos_dim, self.pose_latent_dim) #6,64
        self.firstposeEmbedding = nn.Linear(self.pos_dim, self.pose_latent_dim) #6,64
        # self.audioEmbedding = nn.Linear(self.audio_dim, self.audio_latent_dim) #1024, 256
        self.audioEmbedding = MyResNet34(256, 1)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, batch):
        '''
            x: 6-dim pos, (bs, max_num_frames, 6)
            y: 1024-dim audio embbeding, (bs, max_num_frames, 1024)
        '''

        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs = x.shape[0]
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats) 
        x_ref = x[:,0,:].unsqueeze(dim=1) # The pose information of the first frame(refrence img)
        x = x-x_ref.repeat(1,x.size(1),1) # bs, nf, 6  Obtain the difference from the first frame
        batch['x_delta'] = x
        x_ref = x_ref.permute((1,0,2)) #1, bs, 6
        x = x.permute((1, 0, 2)) #nf, bs, 6
        y = y.permute(1,0,2,3).unsqueeze(dim=2) # nf, bs, 1, 4, 41
        y = rearrange(y, 'f b c x y -> (f b) c x y')
        # embedding of the pose/audio
        x_ref = self.firstposeEmbedding(x_ref).repeat(x.size(0),1,1) # nf, bs, 64
        x = self.poseEmbedding(x) #nf, bs, 64
        y = self.audioEmbedding(y) # nf*bs, 1, 4, 41 -> nf*bs, 256
        y = rearrange(y, '(f b) c -> f b c', b=bs)  # nf, bs, 256
        x = torch.cat([x_ref, x, y],dim=-1) # nf, bs, 64+64+256

        # only use the "average_encoder" mode
        # add positional encoding
        x = self.sequence_pos_encoder(x)
        # transformer layers
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask) #nu_frames, bs, 64+64+256
        # get the average of the output
        # z = final.mean(axis=0) # bs, 64+64+256
        z = final # nf, bs, 64+64+256
        # extract mu and logvar
        mu = self.mu_layer(z) # nf, bs, 256
        logvar = self.sigma_layer(z) # nf, bs, 256

        return {"mu": mu, "logvar": logvar}


class Decoder_TRANSFORMERMEL(nn.Module):
    def __init__(self, modeltype, num_frames, audio_dim=1024, pos_dim=6, pose_latent_dim=64,
                 audio_latent_dim=256, ff_size=128, num_layers=2, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype

        self.pos_dim = pos_dim
        self.num_frames = num_frames
        self.audio_dim = audio_dim
        
        self.pose_latent_dim = pose_latent_dim
        self.audio_latent_dim = audio_latent_dim
        self.latent_dim = self.audio_latent_dim + self.pose_latent_dim*2
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation

        self.firstposeEmbedding = nn.Linear(self.pos_dim, self.pose_latent_dim) #6,64
        # the first version
        # self.audioEmbedding = nn.Linear(self.audio_dim, self.audio_latent_dim) #1024, 256
        # the second version, inspired from audio2head
        self.audioEmbedding = MyResNet34(256, 1)
        self.ztimelinear = nn.Linear(self.audio_latent_dim*2+self.pose_latent_dim, self.pose_latent_dim) #256*2+64,64
        # self.input_feats = self.njoints*self.nfeats

        # # only for ablation / not used in the final model
        # if self.ablation == "zandtime":
        #     self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        # else:
        #     self.actionBiases = nn.Parameter(torch.randn(1024, self.latent_dim))
            # self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # # only for ablation / not used in the final model
        # if self.ablation == "time_encoding":
        #     self.sequence_pos_encoder = TimeEncoding(self.dropout)
        # else:
        #     self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.sequence_pos_encoder = PositionalEncoding(self.pose_latent_dim, self.dropout)
        # self.sequence_pos_encoder = TimeEncoding(self.dropout) #time_encoding
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.pose_latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.pose_latent_dim, self.pos_dim)
        
    def forward(self, batch):
        '''
            z: nf, bs, audio_latent_dim(256)
            y: bs, num_frames, 1024
            mask: bs, num_frames
            lengths: [num_frames,...]
        '''
        x, z, y, mask, lengths = batch["x"], batch["z"], batch["y"], batch["mask"], batch["lengths"]
        bs, nframes = mask.shape
        # first img
        x_ref = x[:,0,:].unsqueeze(dim=1) #bs, 1, 64
        x_ref = self.firstposeEmbedding(x_ref.repeat(1, nframes, 1)) #bs, nf, 64
        y = y.permute(1,0,2,3).unsqueeze(dim=2) # nf, bs, 1, 4, 41
        y = rearrange(y, 'f b c x y -> (f b) c x y')
        y = self.audioEmbedding(y) # nf*bs, 1, 4, 41 -> nf*bs, 256
        y = rearrange(y, '(f b) c -> f b c', b=bs)  # nf, bs, 256
        y = y.permute(1, 0, 2)
        # z = z.unsqueeze(dim=1).repeat(1, nframes, 1) #bs, num_frames, 256
        z = z.permute(1, 0, 2)
        z = torch.cat([x_ref, z, y], dim=-1) # bs, num_frames, 256*2+64
        z = self.ztimelinear(z)
        z = z.permute((1, 0, 2)) # nf, bs, 64
        pose_latent_dim = z.shape[2]
        # z = z[None]  # sequence of size 1

        # # only for ablation / not used in the final model
        # if self.ablation == "zandtime":
        #     yoh = F.one_hot(y, self.num_classes)
        #     z = torch.cat((z, yoh), axis=1)
        #     z = self.ztimelinear(z)
        #     z = z[None]  # sequence of size 1
        # else:
        #     # only for ablation / not used in the final model
        #     if self.ablation == "concat_bias":
        #         # sequence of size 2
        #         z = torch.stack((z, self.actionBiases[y]), axis=0)
        #     else:
        #         # shift the latent noise vector to be the action noise
        #         z = z + self.actionBiases[y.long()] # NEED CHECK
        #         z = z[None]  # sequence of size 1
            
        timequeries = torch.zeros(nframes, bs, pose_latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        # timequeries = self.sequence_pos_encoder(timequeries, mask, lengths) #time_encoding
        
        # # only for ablation / not used in the final model
        # if self.ablation == "time_encoding":
        #     timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        # else:
        #     timequeries = self.sequence_pos_encoder(timequeries)
        
        # num_frames, bs, 64
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output).reshape(nframes, bs, self.pos_dim) # num_frames, bs, 6
        # output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0 #nf, bs, 6
        output = output.permute(1,0,2)#bs, nf, 6
        
        batch["output"] = output
        return batch
