from typing import Union

from pathlib import Path

import torch
import numpy as np
import re
import cv2
import random
from matplotlib import pyplot as plt
from PIL import Image


def readPFM(file: str) -> np.ndarray:
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    flow = np.delete(data, 2, axis=2)
    return flow.astype(np.float32)

# class FlyingChairDataset(torch.utils.data.Dataset):
class DrivingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root: Union[Path, str],
                 transform = None) -> None:

        self.root = Path(root)
        self.root_img = self.root / 'image' 
        self.ids = [o.stem for o in self.root_img.iterdir()]
        # self.ids = list(self.ids)
        self.transform = transform

    def __getitem__(self, idx: int):
        id_ = int(self.ids[idx])

        frame1_path = self.root / 'image' / (f'{id_:0>4}' + '.png')
        frame2_path = self.root / 'image' / (f'{id_+1:0>4}' + '.png')
        optical_flow_path = self.root / 'flow' / ('OpticalFlowIntoFuture_' + f'{id_:0>4}' + '_L.pfm')
        # OpticalFlowIntoFuture_0001_L
        frame1 = cv2.imread(str(frame1_path), )
        frame2 = cv2.imread(str(frame2_path), )
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        optical_flow = readPFM(str(optical_flow_path))

        if self.transform is not None:
            (frame1, frame2), optical_flow = \
                self.transform((frame1, frame2), optical_flow)
            
        return (frame1, frame2), optical_flow

    def __len__(self) -> int:
        return len(self.ids) - 1 # (-1) is bc we have a PAIR of images as inputs

class MonkaaDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir: Union[Path, str], transform=None):
        self.root_dir = Path(root_dir) / 'image'
        self.flow_root_dir = Path(root_dir) / 'flow'
        self.transform = transform
        self.classes, self.classes_comu_len = self._find_classes(self.root_dir)

    def _find_classes(self, dir: Path):
        classes = sorted([d.name for d in dir.iterdir() if d.is_dir()])
        classes_len = [len(list(self.root_dir.glob(f'{d}/*.png'))) for d in classes]
        classes_comu_len = {cls_name: (classes_len[i], np.sum(classes_len[0:i+1], dtype=np.int16)) for i, cls_name in enumerate(classes)}
        # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, classes_comu_len
    
    def __len__(self):
        ext = '*.png'
        n = len(list(self.root_dir.glob(f'**/{ext}')))
        return n

    def __getitem__(self, idx: int):
        ext = '*.png'
        # files = sorted(list(self.root_dir.glob(f'**/{ext}')))
        files = []
        for scene in self.classes:
            addrr = list(self.root_dir.glob(f'{scene}/{ext}'))
            files = files + sorted(addrr)

        file_path = files[idx]
        categ = file_path.parent.name
        cls_len, comu_len = self.classes_comu_len[categ]

        if idx == comu_len - 1: #if the index is for last image of a category, use prev frame as frame 1
            frame1 = cv2.imread(str(files[idx-1]), )
            frame2 = cv2.imread(str(files[idx]), )
            optical_flow_path = self.flow_root_dir / categ / f'OpticalFlowIntoFuture_{files[idx-1].stem}_L.pfm'
        else: #Otherwise use idx for frame 1, so frame 2 is idx+1
            frame1 = cv2.imread(str(files[idx]), )
            frame2 = cv2.imread(str(files[idx+1]), )
            optical_flow_path = self.flow_root_dir / categ / f'OpticalFlowIntoFuture_{files[idx].stem}_L.pfm'

        optical_flow = readPFM(str(optical_flow_path))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            (frame1, frame2), optical_flow = \
                self.transform((frame1, frame2), optical_flow)
            
        return (frame1, frame2), optical_flow
    

if __name__ == "__main__":
    import utils
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # print(os.getcwd())
    data = MonkaaDataset("./Monkaa_cleanpass" , None)
    # data = DrivingDataset('driving_dataset')
    print(len(data))
    # dl = torch.utils.data.DataLoader(data, batch_size = 16, num_workers=2, drop_last=True, shuffle=True)
    # for index, (x, y) in enumerate(dl):
    #     print(type(x[0]))
    #     print(x[0].size())
    print(data.classes_comu_len)
    # data = FlyingChairDataset("./dataset" , None)
    # print(len(data))
    # print(device)
    # dl = torch.utils.data.DataLoader(data, batch_size = 16, num_workers=2, drop_last=True)

    # max_flow = 0
    # BS_flow = []
    # for i in range(len(data)):
    #     _, flow = data[i]
    #     m = np.max(flow)
    #     if m>200:
    #         BS_flow.append((i, m))
    #         print(f'At {i} flow is {m}')
    #     if m>max_flow:
    #         max_flow = m
    #         index = i
    #     if i%100==0:
    #         print(f'checking {i} files is done')
    # print(max_flow)
    # print(BS_flow)

    frames, true_flow = data[873]
    print(frames[0].shape)
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(frames[0])
    plt.subplot(132)
    plt.imshow(frames[1])
    plt.subplot(133)
    plt.imshow(utils.flow_to_image(true_flow))
    plt.show()

    utils.quiver_plot(cv2.resize(true_flow, dsize=(0,0), fx=1/16, fy=1/16) / 16)
    plt.show()
