import random
from typing import Union, Tuple

import torch
import torchvision.transforms.functional as F
import cv2
import torch.nn.functional as FNN
import numpy as np
# from skimage import transform


ImageOrTensor = Union[np.ndarray, torch.Tensor]
Transformed = Tuple[Tuple[ImageOrTensor, ImageOrTensor], 
                    Union[np.ndarray, torch.Tensor]]


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        o = args
        for trf in self.transforms:
            o = trf(*o)
        return o


class Resize(object):

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
    
    def __call__(self, 
                 frames: Tuple[np.ndarray, np.ndarray], 
                 optical_flow: np.ndarray) -> Transformed:
        
        frame1 = F.resize(frames[0], (self.height, self.width) , antialias=True)
        frame2 = F.resize(frames[1], (self.height, self.width), antialias=True)
        # print(self.width, optical_flow.size()[2]) # -> (60,960) for level 0
        optical_flow = F.resize(optical_flow, (self.height, self.width), 
                                antialias=True) / optical_flow.size()[2] * self.width
        # print(self.width, optical_flow.size()[2])
        # optical_flow = FNN.interpolate(optical_flow, size=(self.height, self.width),
        #                              align_corners=True, mode='bilinear') * optical_flow.size()[2] / self.width

        return (frame1, frame2), optical_flow


class RandomRotate(object):

    def __init__(self, minmax: Union[Tuple[int, int], int]) -> None:
        self.minmax = minmax
        if isinstance(minmax, int):
            self.minmax = (-minmax, minmax)
    
    def __call__(self, 
                 frames: Tuple[np.ndarray, np.ndarray], 
                 optical_flow: np.ndarray) -> Transformed:
        angle = random.randint(*self.minmax)
        frame1 = F.rotate(frames[0], angle)
        frame2 = F.rotate(frames[1], angle)
        optical_flow = F.rotate(optical_flow, angle) # this is wrong

        return (frame1, frame2), optical_flow


class Normalize(object):

    def __init__(self, 
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self, 
                 frames: Tuple[torch.Tensor, torch.Tensor], 
                 optical_flow: torch.Tensor) -> Transformed:

        frame1 = F.normalize(frames[0], self.mean, self.std)
        frame2 = F.normalize(frames[1], self.mean, self.std)

        return (frame1, frame2), optical_flow


class ToTensor(object):

    def __call__(self, 
                 frames: Tuple[np.ndarray, np.ndarray], 
                 optical_flow: np.ndarray) -> Transformed:
        
        return ((F.to_tensor(frames[0]), F.to_tensor(frames[1])), 
                 torch.from_numpy(optical_flow).permute(2, 0, 1).float())
