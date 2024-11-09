'''
stage 1: using 0th as the reference, short and fixed clip

with lip loss, 6D pose, conditioned by cross attention

'''
import os
import torch
import torch.nn as nn
import sys
sys.path.append('your/path')
from LFG.modules.generator import Generator
from LFG.modules.bg_motion_predictor import BGMotionPredictor
from LFG.modules.region_predictor import RegionPredictor
from DM_3.modules.video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi import DynamicNfUnet3D, DynamicNfGaussianDiffusion
import yaml
from sync_batchnorm import DataParallelWithCallback
from filter_fourier import *

from DM_v0_loss_exp.modules.util import AntiAliasInterpolation2d
from torchvision import models
import numpy as np
import time
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.fc_attention = nn.Linear(params['dim_attention'], 1)
    
    def forward(self, ctx_val, ctx_key, ctx_mask, ht_query):

        ht_query = self.fc_query(ht_query)

        attention_score = torch.tanh(ctx_key + ht_query[:, None, None, :])
        attention_score = self.fc_attention(attention_score).squeeze(3)
        
        attention_score = attention_score - attention_score.max()
        attention_score = torch.exp(attention_score) * ctx_mask
        attention_score = attention_score / (attention_score.sum(2).sum(1)[:, None, None] + 1e-10)

        ct = (ctx_val * attention_score[:, None, :, :]).sum(3).sum(2)

        return ct, attention_score
        
class Face_loc_Encoder(nn.Module):
    def __init__(self, dim = 1):
        super(Face_loc_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(dim, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        return x

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

class FlowDiffusion(nn.Module):
    def __init__(self, img_size=32, num_frames=40, sampling_timesteps=250,
                 null_cond_prob=0.1,
                 ddim_sampling_eta=1.,
                 dim_mults=(1, 2, 4, 8),
                 is_train=True,
                 use_residual_flow=False,
                 learn_null_cond=False,
                 use_deconv=True,
                 padding_mode="zeros",
                 pretrained_pth="",
                 config_pth=""):
        super(FlowDiffusion, self).__init__()
        self.use_residual_flow = use_residual_flow

        checkpoint = torch.load(pretrained_pth)
        with open(config_pth) as f:
            config = yaml.safe_load(f)

        self.generator = Generator(num_regions=config['model_params']['num_regions'],
                                   num_channels=config['model_params']['num_channels'],
                                   revert_axis_swap=config['model_params']['revert_axis_swap'],
                                   **config['model_params']['generator_params']).cuda()
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()
        self.set_requires_grad(self.generator, False)

        self.region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                                num_channels=config['model_params']['num_channels'],
                                                estimate_affine=config['model_params']['estimate_affine'],
                                                **config['model_params']['region_predictor_params']).cuda()
        self.region_predictor.load_state_dict(checkpoint['region_predictor'])
        self.region_predictor.eval()
        self.set_requires_grad(self.region_predictor, False)

        self.bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                              **config['model_params']['bg_predictor_params'])
        self.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        self.bg_predictor.eval()
        self.set_requires_grad(self.bg_predictor, False)

        self.scales = config['train_params']['scales']
        self.pyramid = ImagePyramide(self.scales, self.generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        # self.vgg_loss_weights = config['train_params']['loss_weights']['rec_vgg']
        # if sum(self.vgg_loss_weights) != 0:
        #     self.vgg = Vgg19()
        #     if torch.cuda.is_available():
        #         self.vgg = self.vgg.cuda()

        self.unet = DynamicNfUnet3D(dim=64,
                           cond_dim=1024 + 6 + 2,
                           cond_aud=1024,
                           cond_pose=6,
                           cond_eye=2,
                           num_frames=num_frames,
                           channels=3 + 256 + 16,
                           out_grid_dim=2,
                           out_conf_dim=1,
                           dim_mults=dim_mults,
                           use_hubert_audio_cond=True,
                           learn_null_cond=learn_null_cond,
                           use_final_activation=False,
                           use_deconv=use_deconv,
                           padding_mode=padding_mode)

        self.diffusion = DynamicNfGaussianDiffusion(
            denoise_fn = self.unet,
            num_frames=num_frames,
            image_size=img_size,
            sampling_timesteps=sampling_timesteps,
            timesteps=1000,  # number of steps
            loss_type='l2',  # L1 or L2
            use_dynamic_thres=True,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta
        )

        self.face_loc_emb = Face_loc_Encoder()

        # training
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()

    def update_num_frames(self, new_num_frames):
        # to update num_frames of Unet3D and GaussianDiffusion
        self.unet.update_num_frames(new_num_frames)
        self.diffusion.update_num_frames(new_num_frames)

    def generate_bbox_mask(self, bbox, size = 32):
        # b = bbox.shape[0]

        b, c, fn = bbox.size()
        bbox = bbox[:,:,0]  # b, c, fn
        bbox[:, :2] = (bbox[:, :2]/bbox[:, 4].unsqueeze(1)) * size  # rescale to 32* 32 for 128, and 64 * 64 for 256
        bbox[:,2:4] = (bbox[:, 2:4]/bbox[:, 5].unsqueeze(1) )* size

        bbox_left_top = bbox[:, :4:2].to(torch.int32)  # left up
        bbox_right_bottom = (bbox[:, 1:4:2] +1).to(torch.int32)  # right down

        # generating 2D index
        row_indices = torch.arange(size).view(1, size, 1).expand(b, size, size).to(torch.uint8).cuda()
        col_indices = torch.arange(size).view(1, 1, size).expand(b, size, size).to(torch.uint8).cuda()

        # set the face bbox as 1, the first channel is y， the second is x
        mask = (row_indices >= bbox_left_top[:, 1].view(b, 1, 1)) & (row_indices <= bbox_right_bottom[:, 1].view(b, 1, 1)) & \
            (col_indices >= bbox_left_top[:, 0].view(b, 1, 1)) & (col_indices <= bbox_right_bottom[:, 0].view(b, 1, 1))
        # mask : b,32,32

        bbox_mask = mask.unsqueeze(1).float()  # b, 1, 32, 32

        return bbox_mask

    def generate_mouth_mask(self, mouth_lmk, origin_size, size = 32):
        b, fn, pn, c = mouth_lmk.size()  # b, fn  12,  2

        origin_size = origin_size.unsqueeze(1)
        mouth_lmk = (mouth_lmk/origin_size) * size

        ld_coner = mouth_lmk.max(dim=-2)[0]
        ru_coner = mouth_lmk.min(dim=-2)[0]

        row_indices = torch.arange(size).view(1, size, 1).expand(b, fn, size, size).to(torch.uint8).cuda()
        col_indices = torch.arange(size).view(1, 1, size).expand(b, fn, size, size).to(torch.uint8).cuda()

        mask = (row_indices >= ru_coner[:,:, 1].view(b, fn, 1, 1)) & (row_indices <= ld_coner[:,:, 1].view(b, fn, 1, 1)) & \
            (col_indices >= ru_coner[:,:, 0].view(b, fn, 1, 1)) & (col_indices <= ld_coner[:,:, 0].view(b, fn, 1, 1))

        bbox_mask = (mask).float()  # b, 1, 32, 32

        return bbox_mask

    def forward(self, real_vid, ref_img, ref_text, ref_pose, ref_eye_blink, bbox, mouth_lmk, is_eval=False, ref_id = 0):  
        if True:
            b,c,f,h,w = real_vid.size()
            real_vid = rearrange(real_vid, 'b c f h w -> (b f) c h w')
            bright = 64. / 255
            contrast = 0.25
            sat = 0.25
            hue = 0.04

            color_jitters = transforms.ColorJitter(hue = (-hue, hue), \
                                                   contrast = (max(0, 1 - contrast), 1 + contrast), 
                                                   saturation = (max(0, 1 - sat), 1 + sat), 
                                                   brightness = (max(0, 1 - bright), 1 + bright))

            # mast have shape :  [..., 1 or 3, H, W]
            real_vid = real_vid/255.  # because the img are floats, so need to scale to 0-1
            real_vid = color_jitters(real_vid)  # shape need be checked
            real_vid = rearrange(real_vid, '(b f) c h w -> b c f h w', b = b, f = f)
            ref_img = real_vid[:,:,ref_id,:,:].clone().detach()

        b, _, nf, H, W = real_vid.size()

        

        ref_pose = ref_pose.squeeze(1).permute(0, 2, 1)[:, :, :-1]
        ref_eye_blink = ref_eye_blink.squeeze(1).permute(0, 2, 1)

        init_pose = ref_pose[:, ref_id].unsqueeze(1).repeat(1, nf, 1)     # b, fn, 7  init state
        init_eye = ref_eye_blink[:, ref_id].unsqueeze(1).repeat(1, nf, 1) # b, fn, 2

        ref_text = torch.concat([ref_text, (ref_pose-init_pose), (ref_eye_blink-init_eye)], dim=-1)

        bbox_mask = self.generate_bbox_mask(bbox, size = real_vid.shape[-1])    # b, 1, 32, 32

        bbox_mask = self.face_loc_emb(bbox_mask)  # conv encoder for face mask

        mouth_mask = self.generate_mouth_mask(mouth_lmk, bbox[:,None,-2:,0], size = real_vid.shape[-1]//4)

        real_grid_list = []
        real_conf_list = []
        real_out_img_list = []
        real_warped_img_list = []
        output_dict = {}
        with torch.no_grad():

            b,c,f,h,w = real_vid.size()
            real_vid_tmp = rearrange(real_vid, 'b c f h w -> (b f) c h w') # real_vid.reshape(b * f, c, h,  w) 
            ref_img_tmp = ref_img.unsqueeze(1).repeat(1,f,1,1,1).reshape(-1, 3, h, w)
            source_region_params = self.region_predictor(ref_img_tmp)
           
            driving_region_params = self.region_predictor(real_vid_tmp)
            bg_params = self.bg_predictor(ref_img_tmp, real_vid_tmp)
            generated = self.generator(ref_img_tmp, source_region_params=source_region_params,
                                        driving_region_params=driving_region_params, bg_params=bg_params)
            output_dict["real_vid_grid"] = rearrange(generated["optical_flow"], '(b f) h w c -> b c f h w', b = b, f = f)
            output_dict["real_vid_conf"] = rearrange(generated["occlusion_map"], '(b f) c h w -> b c f h w', b = b, f = f)
            output_dict["real_out_vid"] = rearrange(generated["prediction"], '(b f) c h w -> b c f h w', b = b, f = f)
            output_dict["real_warped_vid"] = rearrange(generated["deformed"], '(b f) c h w -> b c f h w', b = b, f = f)
        ref_img_fea = generated["bottle_neck_feat"][::f].clone().detach()       #bs, 256, 32, 32
        del real_vid_tmp, ref_img_tmp
        del generated


        if self.is_train:
            if self.use_residual_flow:
                h, w, = H // 4, W // 4
                identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
                output_dict["loss"], output_dict["null_cond_mask"] = self.diffusion(
                    torch.cat((output_dict["real_vid_grid"] - identity_grid,
                               output_dict["real_vid_conf"] * 2 - 1), dim=1),
                    ref_img_fea,
                    bbox_mask,
                    ref_text)
            else:
                output_dict["loss"], output_dict["null_cond_mask"] = self.diffusion(
                    torch.cat((output_dict["real_vid_grid"],
                               output_dict["real_vid_conf"] * 2 - 1), dim=1),
                    ref_img_fea,
                    bbox_mask,
                    ref_text)
            
            pred = self.diffusion.pred_x0
            pred_flow = pred[:, :2, :, :, :]
            pred_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
            # loss_high_freq = hf_loss(fea = pred_flow, mask = self.gaussian_mask.cuda(), dim = 2)
            loss_high_freq = nn.MSELoss(reduce = False)(pred_flow, output_dict["real_vid_grid"]) + nn.MSELoss(reduce = False)(pred_conf, output_dict["real_vid_conf"])
            output_dict['mouth_loss'] = ((output_dict["loss"] * mouth_mask.unsqueeze(1)).sum())/(mouth_mask.sum())
            output_dict["loss"] = output_dict["loss"].mean(1)
            output_dict["floss"] = loss_high_freq.mean(1)

            
            if(is_eval):
                with torch.no_grad():
                    fake_out_img_list = []
                    fake_warped_img_list = []
                    pred = self.diffusion.pred_x0  # bs, 3, nf, 32, 32
                    if self.use_residual_flow:
                        output_dict["fake_vid_grid"] = pred[:, :2, :, :, :] + identity_grid
                    else:
                        output_dict["fake_vid_grid"] = pred[:, :2, :, :, :]  # optical flow predicted by DM_2   bs, 2, nf, 32, 32
                    output_dict["fake_vid_conf"] = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5 # occlusion map  predicted by DM_2   bs, 1, nf, 32, 32
                    for idx in range(nf):
                        fake_grid = output_dict["fake_vid_grid"][:, :, idx, :, :].permute(0, 2, 3, 1) #bs, 32, 32, 2
                        fake_conf = output_dict["fake_vid_conf"][:, :, idx, :, :]                     #bs, 1, 32, 32
                        # predict fake out image and fake warped image
                        generated = self.generator.forward_with_flow(source_image=ref_img,
                                                                        optical_flow=fake_grid,
                                                                        occlusion_map=fake_conf)
                        fake_out_img_list.append(generated["prediction"])
                        fake_warped_img_list.append(generated["deformed"].detach())
                        del generated
                    output_dict["fake_out_vid"] = torch.stack(fake_out_img_list, dim=2)
                    output_dict["fake_warped_vid"] = torch.stack(fake_warped_img_list, dim=2).detach()
                    # output_dict["rec_loss"] = nn.L1Loss(reduce=False)(real_vid, output_dict["fake_out_vid"])
                    # output_dict["rec_warp_loss"] = nn.L1Loss(reduce=False)(real_vid, output_dict["fake_warped_vid"])

                    # b,c,f,h,w = real_vid.size()
                    # real_vid_tensor = real_vid.permute(0,2,1,3,4).reshape(b*f,c,h,w).detach()
                    # fake_out_vid_tensor  = output_dict["fake_out_vid"].permute(0,2,1,3,4).reshape(b*f,c,h,w)

                    

                    # if sum(self.vgg_loss_weights) != 0:
                    #     pyramide_real = self.pyramid(real_vid_tensor)
                    #     pyramide_generated = self.pyramid(fake_out_vid_tensor)
                    #     value_total = 0
                    #     for scale in self.scales:
                    #         x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                    #         y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                    #         for i, weight in enumerate(self.vgg_loss_weights):
                    #             value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    #             value_total += self.vgg_loss_weights[i] * value
                    #     output_dict['rec_vgg_loss'] = value_total
                    # if __debug__:
                    #     end_time = time.time()  # end
                    #     # print(f'forward for eval part time {end_time- start_time}')
                    #     start_time = end_time


        return output_dict

    def sample_one_video(self, real_vid, sample_img, sample_audio_hubert, sample_pose, sample_eye, sample_bbox, cond_scale):
        output_dict = {} 
        sample_img_fea = self.generator.compute_fea(sample_img) # sample_img: bs,3,128,128 sample_img_fea: 1,256,32,32
        bbox_mask = self.generate_bbox_mask(sample_bbox, size = sample_img.shape[-1])

        bbox_mask = self.face_loc_emb(bbox_mask)  # conv encoder for face mask


        ref_pose = sample_pose.permute(0, 2, 1)[:,:,:-1]
        ref_eye_blink = sample_eye.permute(0, 2, 1)

        init_pose = ref_pose[:, 0].unsqueeze(1).repeat(1,ref_pose.shape[1], 1)
        init_eye = ref_eye_blink[:, 0].unsqueeze(1).repeat(1,ref_eye_blink.shape[1], 1) # b, fn, 2

        ref_text = torch.concat([sample_audio_hubert, (ref_pose - init_pose), (ref_eye_blink - init_eye)], dim=-1)

        bs = sample_img_fea.size(0)
        # if cond_scale = 1.0, not using unconditional model
        # pred bs, 3, nf, 32, 32
        pred = self.diffusion.sample(sample_img_fea, bbox_mask, cond=ref_text,
                                     batch_size=bs, cond_scale=cond_scale)
        if self.use_residual_flow:
            b, _, nf, h, w = pred[:, :2, :, :, :].size()
            identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
            output_dict["sample_vid_grid"] = pred[:, :2, :, :, :] + identity_grid
        else:
            output_dict["sample_vid_grid"] = pred[:, :2, :, :, :]  # bs, 2, nf, 32, 32
        output_dict["sample_vid_conf"] = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5  # bs, 1, nf, 32, 32
        nf = output_dict["sample_vid_grid"].size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(nf):
                sample_grid = output_dict["sample_vid_grid"][:, :, idx, :, :].permute(0, 2, 3, 1)
                sample_conf = output_dict["sample_vid_conf"][:, :, idx, :, :]
                # predict fake out image and fake warped image
                generated = self.generator.forward_with_flow(source_image=sample_img,
                                                             optical_flow=sample_grid,
                                                             occlusion_map=sample_conf)
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
        output_dict["sample_out_vid"] = torch.stack(sample_out_img_list, dim=2)
        output_dict["sample_warped_vid"] = torch.stack(sample_warped_img_list, dim=2)

        output_dict["rec_loss"] = nn.L1Loss(reduce=False)(real_vid, output_dict["sample_out_vid"])
        output_dict["rec_warp_loss"] = nn.L1Loss(reduce=False)(real_vid, output_dict["sample_warped_vid"])
        
        # b,c,f,h,w = real_vid[0].unsqueeze(dim=0).size()
        # real_vid_tensor = real_vid[0].unsqueeze(dim=0).permute(0,2,1,3,4).reshape(b*f,c,h,w)
        # fake_out_vid_tensor  = output_dict["sample_out_vid"].permute(0,2,1,3,4).reshape(b*f,c,h,w)

        # pyramide_real = self.pyramid(real_vid_tensor)
        # pyramide_generated = self.pyramid(fake_out_vid_tensor)

        # if sum(self.vgg_loss_weights) != 0:
        #     value_total = 0
        #     for scale in self.scales:
        #         x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
        #         y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

        #         for i, weight in enumerate(self.vgg_loss_weights):
        #             value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
        #             value_total += self.vgg_loss_weights[i] * value
        #     output_dict['rec_vgg_loss'] = value_total

        return output_dict

    def get_grid(self, b, nf, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid.permute(0, 3, 1, 2).unsqueeze(dim=2).repeat(1, 1, nf, 1, 1)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():

                    param.requires_grad = requires_grad


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    bs = 5
    img_size = 128
    num_frames = 10
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand((bs, 3, img_size, img_size), dtype=torch.float32).cuda()
    real_vid = torch.rand((bs, 3, num_frames, img_size, img_size), dtype=torch.float32).cuda()
    model = FlowDiffusion(num_frames=num_frames, use_residual_flow=False, sampling_timesteps=10, dim_mults=(1, 2, 4, 8, 16))
    model.cuda()
    # embedding ref_text
    # cond = bert_embed(tokenize(ref_text), return_cls_repr=model.diffusion.text_use_bert_cls).cuda()

    # to simulate the situation of hubert embedding
    cond = torch.rand((bs,10,1024), dtype=torch.float32).cuda()
    model = DataParallelWithCallback(model)
    output_dict = model.forward(real_vid=real_vid, ref_img=ref_img, ref_text=cond)
    model.module.sample_one_video(sample_img=ref_img[0].unsqueeze(dim=0),
                                  sample_audio_hubert=cond[0].unsqueeze(dim=0),
                                  cond_scale=1.0)
