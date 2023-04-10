
from pathlib import Path
from typing import Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset
import model as SpyModel
from SpyPackage import transforms as OFT
from SpyPackage import dataset, utils
import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(data_root: str, checkpoint_name:str, show_acc=False) -> None:

    tfms = OFT.Compose([
        OFT.ToTensor(),
        # OFT.Normalize(mean=[.485, .406, .456], 
        #               std= [.229, .225, .224])
    ])

    model = SpyModel.SpyNet.from_pretrained(checkpoint_name, map_location=device)
    model.to(device)
    model.eval()
    err = SpyModel.EPELoss()
    if show_acc:
        valid_ds = dataset.MonkaaDataset(data_root, transform=tfms)
        # valid_ds = dataset.DrivingDataset(data_root, transform=tfms)

        train_len = int(len(valid_ds) * 0.9)
        rand_idx = torch.randperm(len(valid_ds)).tolist()
        valid_ds = Subset(valid_ds, rand_idx[train_len:])
        valid_dl = DataLoader(valid_ds,
                                batch_size=2,
                                num_workers=4)
        acc = []
        for i, (x, y) in enumerate(valid_dl):
            x = x[0].to(device), x[1].to(device) # x[0] and x[1] is a torch.Size([Batch_size, 3, H, W])
            y = y.to(device)    
            with torch.no_grad():
                Vk = model(x)

            est_flow_resized = (F.resize(Vk, size=(540, 960), antialias=True) * 960 / Vk.size()[-1]).to(device)
            acc.append(err(est_flow_resized, y).cpu().detach())

        print(np.mean(acc))
        print(np.sum(acc)/len(valid_dl))

    valid_ds = dataset.MonkaaDataset(data_root)
    # valid_ds = dataset.DrivingDataset(data_root)

    o_frames, o_of = valid_ds[480]
    frames, of = tfms(o_frames, o_of)
    frames = [o.unsqueeze(0).to(device) for o in frames]

    with torch.no_grad():
        Vk = model(frames)[0]
    frames = [o.unsqueeze(0).to(device) for o in frames]
    pred_of_im = utils.flow_to_image(Vk)
    true_of = utils.flow_to_image(of)
    true_flow_resized = (F.resize(of, size=(540, 960), antialias=True,) * 960 / of.size()[-1]).to(device)
    est_flow_resized = (F.resize(Vk, size=(540, 960), antialias=True) * 960 / Vk.size()[-1]).to(device)
    print(err(true_flow_resized, est_flow_resized))
    utils.imgplot(o_frames[0], o_frames[1], true_of, pred_of_im)
    plt.figure()
    plt.imshow(pred_of_im)
    plt.show()
    # utils.quiver_plot(true_flow_resized)
    # utils.quiver_plot(est_flow_resized)


if __name__ == "__main__":
    plt.close()
    inference('./Monkaa_cleanpass', './models/tanh_scale_noNormalize20230406-1234/final.pt', show_acc=True)
    # inference('./driving_dataset', './models/tanh_scale_noNormalize20230406-1234/final.pt')
