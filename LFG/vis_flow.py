import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_dense_optical_flow(flow_tensor, save_path):
    flow_np = flow_tensor.cpu().numpy()
    flow_tensor = flow_tensor + 1e-7

    magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)

    # mask = magnitude > 1/64


    magnitude = magnitude # * mask
    angle = np.arctan2(flow_np[1], flow_np[0])

    angle = angle # * mask

    plt.figure()
    plt.imshow(magnitude, cmap='BuPu', alpha=0.8)
    plt.imshow(angle, cmap='hsv', alpha=0.2)
    plt.title('Dense Optical Flow')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def grid2flow(warped_grid, grid_size=64, img_size=256):
    dpi = 1000
    # plt.ioff()
    h_range = torch.linspace(-1, 1, grid_size)
    w_range = torch.linspace(-1, 1, grid_size)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2)
    
    out = warped_grid - grid
    return out

if __name__ == '__main__':
    dense_flow_tensor = torch.zeros(2, 100, 100)
    visualize_dense_optical_flow(dense_flow_tensor, 'test.jpg')