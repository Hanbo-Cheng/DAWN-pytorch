import torch
import torch.fft
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Filtering function: Input optical flow field to filter out high-frequency noise.
def gaussian_pdf(x, mean, std):
        return (1 / (std * torch.sqrt(2 * torch.tensor(3.141592653589793))) *
                torch.exp(-((x - mean) ** 2) / (2 * std ** 2)))

def gaussian_density(length = 20, amplitude = 2, mean = 19, sigma = 3):
    x = torch.arange(0, length, 1.0)
    gaussian = amplitude * torch.exp(-(x - mean)**2 / (2 * sigma**2))
    gaussian = torch.clip(gaussian, max = 1, min = 0)
    return gaussian.cuda()

def fourier_filter(fea):
    L, C , H , W = fea.shape
    mean = 0
    std = 3
    _x = torch.linspace(-10, 10, H)  # Define 128 values within the range of -5 to 5.
    X, Y = torch.meshgrid(_x, _x)  # Generate grid coordinates.
    gaussian_map = (gaussian_pdf(X, mean, std).cuda()) * (gaussian_pdf(Y, mean, std).cuda())
    gaussian_map = gaussian_map.unsqueeze(0).repeat(1, C, 1, 1)

    gaussian_map = torch.clip((gaussian_map)/gaussian_map.max() * 3 , min = 0, max = 1)

    # lowpass_filter = torch.zeros(H,H).cuda()
    # for i in range(H):
    #     for j in range(H):
    #         if np.sqrt((i - H//2)**2 + (j - H//2)**2) <= 10:
    #             lowpass_filter[i, j] = 1

    x = torch.fft.fft2(fea, dim=(-2, -1))
    x_shifted = torch.fft.fftshift(x)  # 1,3,128,128

    x_shifted = x_shifted * gaussian_map# lowpass_filter #  * gaussian_map

    reconstructed_x = torch.fft.ifftshift(x_shifted)
    reconstructed_x = torch.fft.ifft2(reconstructed_x, dim=(-2, -1))
    reconstructed_x = torch.real(reconstructed_x)


    return reconstructed_x

def fourier_filter_1D(fea, dim):
    # idex = freq * L / 25
    L, C , H , W = fea.shape
    mean = 0
    std = 3
    fft_result = torch.fft.rfft(fea, dim=dim)

    # 低通滤波
    cutoff_freq = 10  # 保留前 10 个频率
    # mask = gaussian_density(length = L, mean = 0, sigma = 5, amplitude = 2)[:, None, None, None]
    # fft_result = mask * fft_result
    fft_result[L//4:] = 0  # 设置高频部分为 0

    # 对 H 维度进行逆傅里叶变换
    filtered_tensor = torch.fft.irfft(fft_result,n= L, dim=dim)
    filtered_tensor = torch.real(filtered_tensor)

    return filtered_tensor

def hf_loss(fea, mask, dim):
    mask = 1- mask # gaussian_density(length = L, mean = 0, sigma = 12, amplitude = 2)
    fft_result = torch.fft.rfft(fea, dim=dim)
    fft_result = fft_result * mask 
    fft_result = fft_result.abs()

    return fft_result

def hf_loss_2(fea_x, fea_y, dim):
    '''
    与GT计算频域损失
    '''
    fft_result_x = torch.fft.rfft(fea_x, dim=dim)
    fft_result_y = torch.fft.rfft(fea_y, dim=dim)
    # fft_result = fft_result.abs()
    loss = (fft_result_y - fft_result_x).abs()

    return loss

    

class KalmanFilter1D:
    def __init__(self, A, H, Q, R, x_init, P_init):
        self.A = torch.tensor(A, requires_grad=False)
        self.H = torch.tensor(H, requires_grad=False)
        self.Q = torch.tensor(Q, requires_grad=False)
        self.R = torch.tensor(R, requires_grad=False)
        self.x = torch.tensor(x_init, requires_grad=True)
        self.P = torch.tensor(P_init, requires_grad=True)

    def update(self, z):
        # 预测步骤
        x_pred = self.A * self.x
        P_pred = self.A * self.P * self.A + self.Q

        # 更新步骤
        K = P_pred * self.H / (self.H * P_pred * self.H + self.R)
        self.x = x_pred + K * (z - self.H * x_pred)
        self.P = (1 - K * self.H) * P_pred

        return self.x

def kalman_filter(observations, dim):
    kf = KalmanFilter1D(A=1., H=1., Q=0.01, R=0.1, x_init=0., P_init=1.)
    filtered_values = torch.zeros_like(observations)

    for idx in range(observations.size(dim)):
        obs_slice = tuple(slice(None) if i != dim else idx for i in range(len(observations.size())))
        obs = observations[obs_slice]
        filtered_value = kf.update(obs)
        filtered_values[obs_slice] = filtered_value

    return filtered_values

def naive_filter(fea):
    L, C , H , W = fea.shape
    fea_mask = fea.abs()>(1/64)
    fea = fea*fea_mask
    return fea
# def fourier_filter(x):
#     L, C , H , W = x.shape
#     mean = 0
#     std = 3
#     _x = torch.linspace(-5, 5, H)  # 定义一个范围为-5到5的128个值
#     X, Y = torch.meshgrid(_x, _x)  # Generate grid coordinates.
#     gaussian_map = (gaussian_pdf(X, mean, std).cuda()) * (gaussian_pdf(Y, mean, std).cuda())
#     gaussian_map = gaussian_map.unsqueeze(0).repeat(1, C, 1, 1)

#     gaussian_map = (gaussian_map)/gaussian_map.max()

#     x = torch.fft.fft2(x, dim=(-2, -1))
#     x_shifted = torch.fft.fftshift(x)  # 1,3,128,128

#     x_shifted = x_shifted # * gaussian_map

#     reconstructed_x = torch.fft.ifftshift(x_shifted)
#     reconstructed_x = torch.fft.ifft2(reconstructed_x, dim=(-2, -1))
#     reconstructed_x = torch.abs(reconstructed_x)


#     return reconstructed_x



if __name__ == '__main__':
    # 读取视频
    gd = gaussian_density(length = 20, mean = 0, sigma = 5, amplitude = 2)
    print(gd)
    print(gd[:10])
    # cap = cv2.VideoCapture('your_path/demo/s2_20w_newae_crema_s1_10_s2_11-j-sl-vr-of-tr-rmm-ddim0200_1.00/7_s76_1076_ITH_FEA_XX.mp4')


    

    # # 生成均值为0，标准差为3的高斯概率密度分布张量
    # mean = 0
    # std = 3
    # x = torch.linspace(-5, 5, 128)  # 定义一个范围为-5到5的128个值
    # X, Y = torch.meshgrid(x, x)  # Generate grid coordinates.
    # gaussian_map = gaussian_pdf(X, mean, std) * gaussian_pdf(Y, mean, std)
    # gaussian_map = gaussian_map.unsqueeze(0).repeat(1, 3, 1, 1)

    # gaussian_map = ( gaussian_map)/gaussian_map.max()

    # # 输入数据，假设frames是一个包含L帧RGB图像的numpy数组，形状为(L, 3, H, W)
    # frames = np.random.randint(0, 255, (100, 3, 256, 256)).astype(np.uint8)

    # # 设置输出视频的名称、帧率和分辨率


    # def generate_video(frames):
    #     video_name = 'output_video.avi'
    #     fps = 25
    #     resolution = (128, 128)

    #     # 创建视频写入对象
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     video = cv2.VideoWriter(video_name, fourcc, fps, resolution)

    #     # 逐帧将图像写入视频
    #     for i in range(frames.shape[0]):
    #         frame = frames[i][:,:,:].transpose(1, 2, 0).astype(np.uint8)  # 调整通道顺序(H, W, 3)
    #         video.write(frame)

    #     # 释放资源并保存视频
    #     video.release()
    # # 存储还原后的图像帧
    # reconstructed_frames = []

    # # 循环遍历视频的每一帧
    # while(cap.isOpened()):
    #     ret, frame = cap.read()

    #     if not ret:
    #         break

    #     # 将当前帧转换为 PyTorch 张量
    #     frame = torch.tensor(frame)
    #     frame = frame.permute(2, 0, 1).unsqueeze(0).float()

    #     # 对当前帧进行 2D 傅里叶变换
    #     fft_frame = torch.fft.fft2(frame, dim=(-2, -1))
    #     fft_frame_shifted = torch.fft.fftshift(fft_frame)  # 1,3,128,128

    #     # 将频域展开形式还原回图像
    #     # fft_frame_shifted = fft_frame_shifted * gaussian_map

    #     reconstructed_frame = torch.fft.ifftshift(fft_frame_shifted)
    #     reconstructed_frame = torch.fft.ifft2(reconstructed_frame, dim=(-2, -1))
    #     reconstructed_frame = torch.abs(reconstructed_frame)

    #     # 将还原后的图像帧添加到列表中
    #     reconstructed_frames.append(reconstructed_frame)

    # # 将还原后的图像帧转换为数组
    # reconstructed_frames = torch.cat(reconstructed_frames, dim=0)

    # # 将还原后的图像帧转换为 numpy 数组
    # reconstructed_frames = (reconstructed_frames).to(torch.int32)
    # reconstructed_frames = reconstructed_frames.squeeze(1).numpy()

    # # 显示还原后的视频

    # generate_video(reconstructed_frames)
