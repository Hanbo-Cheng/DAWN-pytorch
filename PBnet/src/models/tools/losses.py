import torch
from einops import rearrange
import torch.nn.functional as F
from .hessian_penalty import hessian_penalty
from .mmd import compute_mmd
from .ssim_loss import ssim
from .normalize_data import normalize_data

def compute_rc_loss(model, batch):
    # x = batch["x"] #bs, nf, 6
    x_delta = batch["x_delta"]
    output = batch["output"] #bs, nf, 6
    mask = batch["mask"] #bs, nf

    # gtmasked = x[mask]
    gtmasked = x_delta[mask]
    outmasked = output[mask]
    
    # loss is large in the beginning
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_reg_loss(model, batch):
    # x = batch["x"] #bs, nf, 6
    x_delta = batch["x_delta"]
    mask = batch["mask"] #bs, nf
    x_1 = x_delta[:,:-1]    
    x_2 = x_delta[:,1:]

    # gtmasked = x[mask]
    
    
    # loss is large in the beginning
    loss = F.mse_loss(x_1, x_2, reduction='mean')
    return loss

def compute_rc_weight_loss(model, batch):
    x = batch["x"] #bs, nf, 6
    x_delta = batch["x_delta"]
    output = batch["output"] #bs, nf, 6
    mask = batch["mask"] #bs, nf

    # gtmasked = x[mask] #bs*nf, 6
    gtmasked = x_delta[mask] #bs*nf, 6
    outmasked = output[mask] #bs*nf, 6
    if x.size(2) == 6:
        weights = torch.tensor([3, 3, 3, 1, 1, 1], dtype=torch.float32).cuda()
    elif x.size(2) == 7:
        weights = torch.tensor([3, 3, 3, 1, 1, 1, 0.5], dtype=torch.float32).cuda()
    elif x.size(2) == 8:
        weights = torch.tensor([3, 3, 3, 0, 0, 0, 3, 3], dtype=torch.float32).cuda()
    else:
        weights = torch.ones(x.size(2), dtype=torch.float32).cuda()
    weights = weights.unsqueeze(0)
   
    # loss is large in the beginning
    loss = F.mse_loss(gtmasked*weights, outmasked*weights, reduction='mean')

    return loss


def compute_hp_loss(model, batch):
    loss = hessian_penalty(model.return_latent, batch, seed=torch.random.seed())
    return loss


def compute_kl_loss(model, batch):
    # mu, logvar: bs, 256
    mu, logvar = batch["mu"], batch["logvar"]
    loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return loss

def compute_ssim_loss(model, batch):
    x = batch["x"] #bs, nf, 6
    x_ref = x[:,0,:].unsqueeze(dim=1) #bs, 1, 64
    bs = x_ref.shape[0]
    mask = batch["mask"] #bs, nf

    x_delta = batch["x_delta"]
    output = batch["output"]
    loss = ssimnorm_loss(x_delta, output, mask, bs)


    return loss

def ssimnorm_loss(x, output, mask, bs):
    min_vals = min(x.min(),output.min())
    max_vals = max(x.max(),output.max())
    x_norm = normalize_data(x, min_vals, max_vals)
    out_norm = normalize_data(output, min_vals, max_vals)
    gtmasked = x_norm[mask] #bs*nf, 6
    outmasked = out_norm[mask] #bs*nf, 6
    gtmasked = rearrange(gtmasked, '(b f) c -> b f c', b=bs)
    outmasked = rearrange(outmasked, '(b f) c -> b f c', b=bs)
    gtmasked = gtmasked.unsqueeze(dim=1) # b 1 f c
    outmasked = outmasked.unsqueeze(dim=1) # b 1 f c
    loss = 1-ssim(gtmasked, outmasked, val_range=1, window_size=3)
    return loss

def ssimnorm_self_loss(x, output, mask, bs):
    x_norm = normalize_data(x, x.min(), x.max())
    out_norm = normalize_data(output, output.min(),output.max())
    gtmasked = x_norm[mask] #bs*nf, 6
    outmasked = out_norm[mask] #bs*nf, 6
    gtmasked = rearrange(gtmasked, '(b f) c -> b f c', b=bs)
    outmasked = rearrange(outmasked, '(b f) c -> b f c', b=bs)
    gtmasked = gtmasked.unsqueeze(dim=1) # b 1 f c
    outmasked = outmasked.unsqueeze(dim=1) # b 1 f c
    loss = 1-ssim(gtmasked, outmasked, val_range=1, window_size=5)
    return loss

def ssim255_loss(x, output, mask, bs):
    gtmasked = x[mask] #bs*nf, 6
    outmasked = output[mask] #bs*nf, 6

    # add 128 to ensue input range is 0-255
    gtmasked = rearrange(gtmasked, '(b f) c -> b f c', b=bs)+128
    outmasked = rearrange(outmasked, '(b f) c -> b f c', b=bs)+128

    gtmasked = gtmasked.unsqueeze(dim=1) # b 1 f c
    outmasked = outmasked.unsqueeze(dim=1) # b 1 f c

    loss = 1-ssim(gtmasked, outmasked, val_range=255, window_size=5)
    return loss

def comput_var_loss(model, batch):
    output = batch["output"] #bs, nf, 6
    mask = batch["mask"] #bs, nf
    outmasked = output[mask] #bs*nf, 6

    batch_size, num_frames, dim = output.size()
    outmasked = rearrange(outmasked, '(b f) c -> b f c', b=batch_size)
    variance_loss = 0
    zero_loss = torch.tensor(0)

    for b in range(batch_size):
        for d in range(dim):
            dimension_output = outmasked[b, :, d]  # shape: (bs, nf)
            frame_variance = torch.var(dimension_output)
            variance_loss += frame_variance
    variance_loss /= (batch_size * dim)
    if 3>variance_loss>0:
        return variance_loss
    else:
        return zero_loss

def compute_mmd_loss(model, batch):
    z = batch["z"]
    true_samples = torch.randn(z.shape, requires_grad=False, device=model.device)
    loss = compute_mmd(true_samples, z)
    return loss


_matching_ = {"rc": compute_rc_loss, "rcw": compute_rc_weight_loss,
              "kl": compute_kl_loss, "hp": compute_hp_loss,
              "mmd": compute_mmd_loss, "ssim": compute_ssim_loss,
              "var": comput_var_loss, 'reg': compute_reg_loss}

# _matching_ = {"rc": compute_rc_loss, "kl": compute_kl_loss, "hp": compute_hp_loss,
#               "mmd": compute_mmd_loss, "rcxyz": compute_rcxyz_loss,
#               "vel": compute_vel_loss, "velxyz": compute_velxyz_loss}


def get_loss_function(ltype):
    return _matching_[ltype]


def get_loss_names():
    return list(_matching_.keys())
