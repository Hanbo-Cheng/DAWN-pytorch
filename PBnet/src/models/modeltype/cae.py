import torch
import torch.nn as nn

from ..tools.losses import get_loss_function
import torch.nn.functional as F
# from ..rotation2xyz import Rotation2xyz



class CAE(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # self.outputxyz = outputxyz
        
        self.lambdas = lambdas
        
        self.latent_dim = latent_dim
        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        self.device = device
        # self.translation = translation
        # self.jointstype = jointstype
        # self.vertstrans = vertstrans
        
        self.losses = list(self.lambdas) + ["mixed"]


        # self.rotation2xyz = Rotation2xyz(device=self.device)
        # self.param2xyz = {"pose_rep": self.pose_rep,
        #                   "glob_rot": self.glob_rot,
        #                   "glob": self.glob,
        #                   "jointstype": self.jointstype,
        #                   "translation": self.translation,
        #                   "vertstrans": self.vertstrans}
        
    # def rot2xyz(self, x, mask, **kwargs):
    #     kargs = self.param2xyz.copy()
    #     kargs.update(kwargs)
    #     return self.rotation2xyz(x, mask, **kargs)
    
    def forward(self, batch):
        # if self.outputxyz:
        #     batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        # elif self.pose_rep == "xyz":
        #     batch["x_xyz"] = batch["x"]
        
        # encode
        batch.update(self.encoder(batch))
        # decode
        batch.update(self.decoder(batch))

        # # if we want to output xyz
        # if self.outputxyz:
        #     batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        # elif self.pose_rep == "xyz":
        #     batch["output_xyz"] = batch["output"]
        return batch

    

    def compute_loss(self, batch, epoch = 0):
        mixed_loss = 0
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            if 'kl' in ltype:
                if epoch < 1e4 and epoch != 0: 
                    lam = 0
                elif epoch != 0:
                    lam = lam * max(epoch - 1e4, 7e4) / 7e4
            mixed_loss += loss*lam
            losses[ltype] = loss.item()
        
        # D_loss, G_loss = self.calculate_GAN_loss(batch)
        # mixed_loss += G_loss * 0.7
        # losses['GAN_D'] = D_loss
        # losses['GAN_G'] = D_loss
        losses["mixed"] = mixed_loss.item()
        return mixed_loss, losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]
        
        batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]
        
        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]
            
    def generate(self, pose, audio, durations,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1):
        '''
            audio: hubert embeddbing, (bs, fn, 1024)
            durations: different num_frames, (bs, )
        '''
        # if nspa is None:
        #     nspa = 1
        bs = len(audio)
            
        # y = audio.to(self.device).repeat(nspa)  # (view(nspa, nats))
        x = pose.to(self.device)
        y = audio.to(self.device) 

        if len(durations.shape) == 1:
            lengths = durations.to(self.device)
        else:
            lengths = durations.to(self.device).reshape(y.shape)
        
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(audio[0].shape[0], bs, self.latent_dim, device=self.device)
        # z = torch.randn(1, bs, self.latent_dim, device=self.device).repeat(audio[0].shape[0], 1, 1)
        
        # if noise_same_action == "random":
        #     if noise_diff_action == "random":
        #         z = torch.randn(nspa*bs, self.latent_dim, device=self.device)
        #     elif noise_diff_action == "same":
        #         z_same_action = torch.randn(nspa, self.latent_dim, device=self.device)
        #         z = z_same_action.repeat_interleave(bs, axis=0)
        #     else:
        #         raise NotImplementedError("Noise diff action must be random or same.")
        # elif noise_same_action == "interpolate":
        #     if noise_diff_action == "random":
        #         z_diff_action = torch.randn(bs, self.latent_dim, device=self.device)
        #     elif noise_diff_action == "same":
        #         z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(bs, 1)
        #     else:
        #         raise NotImplementedError("Noise diff action must be random or same.")
        #     interpolation_factors = torch.linspace(-1, 1, nspa, device=self.device)
        #     z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa*bs, -1)
        # elif noise_same_action == "same":
        #     if noise_diff_action == "random":
        #         z_diff_action = torch.randn(bs, self.latent_dim, device=self.device)
        #     elif noise_diff_action == "same":
        #         z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(bs, 1)
        #     else:
        #         raise NotImplementedError("Noise diff action must be random or same.")
        #     z = z_diff_action.repeat((nspa, 1))
        # else:
        #     raise NotImplementedError("Noise same action must be random, same or interpolate.")

        batch = {"x": x,"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)
        
        # if self.outputxyz:
        #     batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        # elif self.pose_rep == "xyz":
        #     batch["output_xyz"] = batch["output"]
        
        return batch
    
    def return_latent(self, batch, seed=None):
        return self.encoder(batch)["z"]
