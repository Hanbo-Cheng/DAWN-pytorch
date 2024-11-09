import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 0.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, val_range = 1, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = (0.01*val_range)**2
    C2 = (0.03*val_range)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, val_range=1, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, val_range, size_average)

def read_pose_from_txt(file_path):
    data = np.loadtxt(file_path)
    return data

if __name__ == "__main__":
    pose1 = read_pose_from_txt('/train20/intern/permanent/lmlin2/ReferenceCode/ACTOR-master/exps_delta_pose/HDTF_nf40_kl1_ssim1_128_w5_1w/nofinetune/3500/eval_gt/0/RD_Radio14_000_522_gt')
    pose2 = read_pose_from_txt('/train20/intern/permanent/lmlin2/ReferenceCode/ACTOR-master/exps_delta_pose/HDTF_nf40_kl1_ssim1_128_w5_1w/nofinetune/3500/eval_pred/0/RD_Radio14_000_522')

    pose1_tensor = torch.tensor(pose1).unsqueeze(0).unsqueeze(0).float()[:,:,:,:-1]
    pose2_tensor = torch.tensor(pose2).unsqueeze(0).unsqueeze(0).float()[:,:,:,:-1]
    # pose1_tensor = torch.tensor(pose1).unsqueeze(0).unsqueeze(0).float()[:,:,:,:-1]+128
    # pose2_tensor = torch.tensor(pose2).unsqueeze(0).unsqueeze(0).float()[:,:,:,:-1]+128

    pose1_tensor = (pose1_tensor - pose1_tensor.min()) / (pose1_tensor.max() - pose1_tensor.min())
    pose2_tensor = (pose2_tensor - pose2_tensor.min()) / (pose2_tensor.max() - pose2_tensor.min())

    ssim_loss = 1-ssim(pose1_tensor, pose2_tensor, window_size=3, val_range=1)

    print(ssim_loss)
