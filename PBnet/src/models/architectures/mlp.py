import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

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


class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')
        
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
    
class ResUnet(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=(2,1), padding=1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=(2,1), padding=1)

        self.bridge = ResidualConv(filters[2], filters[3], stride=(2,1), padding=1)

        self.upsample_1 = Upsample(filters[3], filters[3], kernel=(2,1), stride=(2,1))
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1)

        self.upsample_2 = Upsample(filters[2], filters[2], kernel=(2,1), stride=(2,1))
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1)

        self.upsample_3 = Upsample(filters[1], filters[1], kernel=(2,1), stride=(2,1))
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)

        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output

class Encoder_MLP(nn.Module):
    def __init__(self, modeltype, num_frames, audio_dim=1024, pos_dim=7, pose_latent_dim=64,
                 audio_latent_dim=256, ff_size=128, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.resunet = ResUnet()
        self.audio_latent_dim = audio_latent_dim
        # self.num_classes = num_classes
        self.seq_len = num_frames
        self.pose_latent_dim = pose_latent_dim

        self.MLP = nn.Sequential()
        layer_sizes = [pos_dim + self.seq_len * pos_dim + self.seq_len * self.audio_latent_dim, ff_size]
        # layer_sizes[0] = self.audio_latent_dim + self.pose_latent_dim*2
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], ff_size)
        self.linear_logvar = nn.Linear(layer_sizes[-1], ff_size)
        self.linear_audio = nn.Linear(audio_dim, self.audio_latent_dim)

        # self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch):
        # class_id = batch['class']
        pose_motion_gt_ori = batch["x"]                             #bs seq_len 6
        ref = pose_motion_gt_ori[:,0,:]                            #bs 6
        batch['x_delta'] = pose_motion_gt_ori - ref[:,None,:]
        pose_motion_gt = batch['x_delta']
        bs = pose_motion_gt_ori.shape[0]
        audio_in = batch["y"]                  # bs seq_len audio_emb_in_size

        #pose encode
        pose_emb = self.resunet(pose_motion_gt.unsqueeze(1))          #bs 1 seq_len 6 
        pose_emb = pose_emb.reshape(bs, -1)                    #bs seq_len*6

        #audio mapping
        # print(audio_in.shape)
        audio_out = self.linear_audio(audio_in)                # bs seq_len audio_emb_out_size
        audio_out = audio_out.reshape(bs, -1)

        # class_bias = self.classbias[class_id]                  #bs latent_size
        x_in = torch.cat([ref, pose_emb, audio_out], dim=-1) #bs seq_len*(audio_emb_out_size+6)+latent_size
        x_out = self.MLP(x_in)

        mu = self.linear_means(x_out)
        logvar = self.linear_means(x_out)                      #bs latent_size 

        batch.update({'mu':mu, 'logvar':logvar})
        return batch


class Decoder_MLP(nn.Module):
    def __init__(self, modeltype, num_frames, audio_dim=1024, pos_dim=7, pose_latent_dim=64,
                 audio_latent_dim=256, ff_size=128, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.resunet = ResUnet()
        # self.num_classes = num_classes
        self.seq_len = num_frames

        self.MLP = nn.Sequential()
        self.audio_latent_dim = audio_latent_dim
        layer_sizes = [ff_size, self.seq_len * pos_dim]
        input_size = ff_size + self.seq_len*audio_latent_dim + pos_dim
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        
        self.pose_linear = nn.Linear(pos_dim, pos_dim)
        self.linear_audio = nn.Linear(audio_dim, audio_latent_dim)

        # self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch):

        z = batch['z']                                          #bs latent_size
        bs = z.shape[0]
        pose_motion_gt = batch["x"]                             #bs seq_len 6
        ref = pose_motion_gt[:,0,:]    
        # class_id = batch['class']                           #bs 6
        audio_in = batch['y']                           # bs seq_len audio_emb_in_size
        #print('audio_in: ', audio_in[:, :, :10])

        audio_out = self.linear_audio(audio_in)                 # bs seq_len audio_emb_out_size
        #print('audio_out: ', audio_out[:, :, :10])
        audio_out = audio_out.reshape([bs, -1])                 # bs seq_len*audio_emb_out_size
        # class_bias = self.classbias[class_id]                   #bs latent_size

        z = z # + class_bias
        x_in = torch.cat([ref, z, audio_out], dim=-1)
        x_out = self.MLP(x_in)                                  # bs layer_sizes[-1]
        x_out = x_out.reshape((bs, self.seq_len, -1))

        #print('x_out: ', x_out)

        pose_emb = self.resunet(x_out.unsqueeze(1))             #bs 1 seq_len 6

        pose_motion_pred = self.pose_linear(pose_emb.squeeze(1))       #bs seq_len 6

        pose_motion_pred = pose_motion_pred

        batch.update({'output':pose_motion_pred})
        return batch

