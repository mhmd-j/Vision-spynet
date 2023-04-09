from typing import Union
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
IMAGE_SIZE_ORIGINAL = np.array([540, 960], int)
IMAGE_LEVEL_SIZE = [(34, 60), (68, 120), (136, 240), (272, 480), (544, 960)]

def flow_to_image(flow: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert flow into middlebury color code image

    Parameters
    ----------
    flow: np.ndarray
        Optical flow map
    
    Returns
    -------
    np.ndarray optical flow image in middlebury color
    """
    if torch.is_tensor(flow):
        flow = flow.cpu().detach().permute(1, 2, 0).numpy()

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel() -> np.ndarray:
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def warp(image: torch.Tensor, 
         optical_flow: torch.Tensor,
         device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """the function creates a grid of coordinates representing the pixels in the input image. 
    It then modifies these coordinates by adding the corresponding motion vectors from 
    the optical flow tensor, and uses the resulting grid to apply a spatial transformation 
    to the input image..

    Args:
        image (torch.Tensor): a bach of images of size (batch, channel, height, width)
        optical_flow (torch.Tensor): batch of flow vectors (batch, 2, height, width)
        device (torch.device, optional): device. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: warped images (batch, channel, height, width)
    """

    b, c, im_h, im_w = image.size() 
    
    hor = torch.linspace(-1.0, 1.0, im_w).view(1, 1, 1, im_w)
    hor = hor.expand(b, -1, im_h, -1)

    vert = torch.linspace(-1.0, 1.0, im_h).view(1, 1, im_h, 1)
    vert = vert.expand(b, -1, -1, im_w)

    grid = torch.cat([hor, vert], 1).to(device) # shape of (b, 2, im_h, im_w)
    optical_flow = torch.cat([
        optical_flow[:, 0:1, :, :] / ((im_w - 1.0) / 2.0), 
        optical_flow[:, 1:2, :, :] / ((im_h - 1.0) / 2.0)], dim=1)

    # Channels last (which corresponds to optical flow vectors coordinates)
    grid = (grid + optical_flow).permute(0, 2, 3, 1) #required by F.grid_sample
    return F.grid_sample(image, grid=grid, padding_mode='border', 
                         align_corners=True)

def imgplot(frame1: Union[torch.tensor, np.ndarray],
            frame2: Union[torch.tensor, np.ndarray],
            flow_true: Union[torch.tensor, np.ndarray],
            flow_pred: Union[torch.tensor, np.ndarray]):
    plt.figure(figsize=(10, 4))

    plt.subplot(221)
    plt.title('Predictions')
    plt.imshow(flow_pred)
    plt.axis('off')

    plt.subplot(222)
    plt.title('Ground Truth Flow')
    plt.imshow(flow_true)
    plt.axis('off')

    plt.subplot(223)
    plt.title('Ground Truth Frame 1')
    if torch.is_tensor(frame1):
        frame1 = frame1.cpu().detach().permute(1, 2, 0).numpy()
    plt.imshow(frame1)
    plt.axis('off')

    plt.subplot(224)
    plt.title('Ground Truth Frame 2')
    if torch.is_tensor(frame2):
        frame2 = frame2.cpu().detach().permute(1, 2, 0).numpy()
    plt.imshow(frame2)
    plt.axis('off')
    plt.show()

def quiver_plot(flow):
    # Extract the X and Y components of the optical flow vectors
    if torch.is_tensor(flow):
        flow = flow.cpu().detach().permute(1, 2, 0).numpy()
    # Create a grid of coordinates for the quiver plot
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    x, y = np.meshgrid(np.arange(0, u.shape[1]), np.arange(0, u.shape[0]))

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.quiver(x, y, u, v)
    ax.set_xlim(0, u.shape[1])
    ax.set_ylim(0, u.shape[0])

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Optical Flow Vector Field')
    ax.axis('off')
    plt.show()

class GConf(object):
    def __init__(self, level: int, max_g_levels: int = 4) -> None:
        assert level >= 0 and level <= max_g_levels
        # self.base_conf = IMAGE_SIZE_ORIGINAL
        # self.scale = 2 ** (max_g_levels - level)
        self.img_size = IMAGE_LEVEL_SIZE[level]

    @property
    def image_size(self):
        # result = self.base_conf / self.scale
        result = self.img_size
        return result
    
if __name__ == "__main__":
    conf = GConf(3).image_size
    print(conf)
    print(type(conf))