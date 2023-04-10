from pathlib import Path
from typing import Sequence, Tuple, Type
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpyPackage import utils

MAX_G_LEVELS = 4
# we need to have G number of the following class
class SpyNetUnit(nn.Module):

    def __init__(self, input_channels: int = 8):
        super(SpyNetUnit, self).__init__()

        #conv layers are structured based on the paper
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),  # added dropout layer

            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),  # added dropout layer

            nn.Conv2d(64, 32, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),  # added dropout layer

            nn.Conv2d(32, 16, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),  # added dropout layer

            nn.Conv2d(16, 2, kernel_size=7, padding=3, stride=1),
            nn.Tanh(),
            LastLayer(),
            # LastLimit(),
            )
            
    def forward(self, 
                frames: Tuple[torch.Tensor, torch.Tensor], 
                optical_flow: torch.Tensor = None,
                upsample_optical_flow: bool = True) -> torch.Tensor:
        """_summary_

        Args:
            frames (Tuple[torch.Tensor, torch.Tensor]): first and second frames
            optical_flow (torch.Tensor, optional): u, v optical flow. Defaults to None.
            upsample_optical_flow (bool, optional): True is upsampling is required. Defaults to True.

        Returns:
            torch.Tensor: _description_
        """
        f_frame, s_frame = frames

        if optical_flow is None:
            # If optical flow is None (k = 0) then create empty one having the
            # same size as the input frames, therefore there is no need to 
            # upsample it later
            upsample_optical_flow = False
            b, c, h, w = f_frame.size()
            optical_flow = torch.zeros(b, 2, h, w, device=s_frame.device)

        if upsample_optical_flow is True:
            optical_flow = F.interpolate(
                optical_flow, scale_factor=2, align_corners=True, 
                mode='bilinear') * 2

        s_frame = utils.warp(s_frame, optical_flow, s_frame.device)
        s_frame = torch.cat([s_frame, optical_flow], dim=1) #concat over channel
        
        inp = torch.cat([f_frame, s_frame], dim=1) #concat over channel
        # print(type(inp), inp.size())

        # b, c, im_h, im_w = f_frame.size() 
        prediction = self.model(inp)
        # prediction = torch.cat([prediction[:, 0:1, :, :]*im_w, 
        #                         prediction[:, 1:2, :, :]*im_h], axis=1)
        return prediction


class LastLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat([x[:, 0:1, :, :]*x.size()[3]/2, 
                       x[:, 1:2, :, :]*x.size()[2]/2], axis=1)
        return x
    
class LastLimit(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = torch.cat([torch.clamp(x[:, 0:1, :, :], min=-x.size()[3], max=x.size()[3]),
                       torch.clamp(x[:, 1:2, :, :], min=-x.size()[2], max=x.size()[2])], axis=1)
                       
        return x

class SpyNet(nn.Module):

    def __init__(self, units: Sequence[SpyNetUnit] = None, levels_no: int = None):
        super(SpyNet, self).__init__()
        
        self.levels_no = levels_no
        if units is not None and levels_no is not None:
            assert len(units) == levels_no

        if units is None and levels_no is None:
            raise ValueError('At least one argument (units or levels_no) must be' 
                             'specified')

        if units is not None:
            self.units = nn.ModuleList(units)
            self.levels_no = len(units)
        else:
            units = [SpyNetUnit() for _ in range(levels_no)]
            self.units = nn.ModuleList(units)

    def forward(self, 
                frames: Tuple[torch.Tensor, torch.Tensor],
                limit_k: int = -1) -> torch.Tensor:
        """
        Forward path

        Args:
            frames (Tuple[torch.Tensor, torch.Tensor]): each frame batch [BATCH, 3, HEIGHT, WIDTH] Highest resolution frames.
            limit_k (int, optional): _description_. number of levels which user wants to perform the algorithm

        Returns:
            torch.Tensor: _description_
        """
        if limit_k == -1:
            units = self.units
        else:
            units = self.units[:limit_k]
        Vk_1 = None

        for k, G in enumerate(self.units): #G is the neural network - paper naming convention
            im_size = utils.GConf(k).image_size
            x1 = F.interpolate(frames[0], size=im_size, mode='bilinear',
                               align_corners=True)
            x2 = F.interpolate(frames[1], size=im_size, mode='bilinear',
                               align_corners=True)

            if Vk_1 is not None: # Upsample the previous optical flow
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, align_corners=True, 
                    mode='bilinear') * 2.0
            # compute residual flow
            Vk = G((x1, x2), Vk_1, upsample_optical_flow=False) # perform forward pass on the associated SpyNetUnit 
            # plt.figure()
            # pred_of_im = utils.flow_to_image(Vk[0])
            # plt.imshow(pred_of_im)
            # utils.quiver_plot(Vk[0])
            # compute actual flow
            Vk_1 = Vk + Vk_1 if Vk_1 is not None else Vk # prepare Vk_1 for the next(higher) level
        
        return Vk_1

    @classmethod
    def from_pretrained(cls: Type['SpyNet'], 
                        name: str, 
                        map_location: torch.device = torch.device('cpu'),
                        dst_file: str = None) -> 'SpyNet':
        
        def get_model(path: str) -> 'SpyNet':
            checkpoint = torch.load(path, 
                                    map_location=map_location)
            k = len(checkpoint) // 30 

            instance = cls(levels_no=k)
            instance.load_state_dict(checkpoint, strict=False)
            instance.to(map_location)
            return instance

        if Path(name).exists():
            return get_model(str(name))
        else:
            raise ValueError(f'The name {name} is not available. ')
        
        return get_model(str(dst_file))


class EPELoss(torch.nn.Module):

    def __init__(self):
        super(EPELoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = (target - pred).pow(2).sum().sqrt()/target.size()[-1]/target.size()[-2]
        return dist.mean()
    
