import datetime
from pathlib import Path
import numpy as np
from typing import Tuple, Sequence
import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import model as SpyModel
from SpyPackage import transforms as OFT
from SpyPackage import dataset, utils


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# AVAILABLE_PRETRAINED = ['sentinel', 'kitti', 'flying-chair', 'none']
writer = SummaryWriter()


def train_one_epoch(dl: DataLoader,
                    optimizer: torch.optim.AdamW,
                    criterion_fn: torch.nn.Module,
                    Gk: torch.nn.Module, 
                    prev_pyramid: torch.nn.Module = None, 
                    print_freq: int = 100,
                    header: str = ''):
    """Train one epoch for only one level of any pyramid

    Args:
        dl (DataLoader): Dataset
        optimizer (torch.optim.AdamW): optimizer for the NN
        criterion_fn (torch.nn.Module): loss function
        Gk (torch.nn.Module): Network of the level G_k of the pyramid
        prev_pyramid (torch.nn.Module, optional): NN of the previous(smaller) pyramid level . Defaults to None.
        print_freq (int, optional): Printing period. Defaults to 100.
        header (str, optional): header for printing. Defaults to ''.
    """
    Gk.train()
    running_loss = 0.

    if prev_pyramid is not None:
        prev_pyramid.eval()

    for i, (x, y) in enumerate(dl):
        x = x[0].to(device), x[1].to(device) # x[0] and x[1] is a torch.Size([Batch_size, 3, H, W])
        y = y.to(device) # y is a torch.Size([Batch_size, 2, H, W])

        if prev_pyramid is not None:
            with torch.no_grad():
                Vk_1 = prev_pyramid(x)
                Vk_1 = F.interpolate(Vk_1, scale_factor=2, 
                                     mode='bilinear', align_corners=True) * 2.0
        else:
            Vk_1 = None
        # print('x_1 size', x[1].size())
        # print('y size', y.size())
        predictions = Gk(x, Vk_1, upsample_optical_flow=False) # Vk_1 = V_{k-1}
        # print('pred', predictions.size())
        # print('y.size: ', y.size())
        if Vk_1 is not None:
            y = y - Vk_1

        loss = criterion_fn(y, predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer.add_scalar('loss/'+header, loss.item(), i)

        running_loss += loss.item()

        if (i + 1) % print_freq == 0:
            loss_mean = running_loss / i
            print(f'{header} [{i}/{len(dl)}] loss {loss_mean:.4f}')
            # writer.add_scalar('loss/'+header, loss.item(), i)

    loss_mean = running_loss / len(dl)
    print(f'{header} loss {loss_mean:.4f}')
    writer.add_scalar(f'lossOverEpoch/{header.split()[0]}', loss_mean, header.split()[-1])


def load_data(root: str, k: int = 0) -> Tuple[Subset, Subset]:
    train_tfms = OFT.Compose([
        OFT.ToTensor(),
        OFT.Resize(*utils.GConf(k).image_size),
        # OFT.RandomRotate(17),
        # OFT.Normalize(mean=[.485, .406, .456], 
        #               std= [.229, .225, .224])
    ]) 

    valid_tfms = OFT.Compose([
        OFT.ToTensor(),
        # OFT.Resize(*utils.GConf(k).image_size),
        # OFT.Normalize(mean=[.485, .406, .456], 
        #               std= [.229, .225, .224])
    ])
    
    train_ds = dataset.MonkaaDataset(root, transform=train_tfms)
    valid_ds = dataset.MonkaaDataset(root, transform=valid_tfms)
    # randomly split the dataset to two subset with 90-10 ratio for training adn validation
    train_len = int(len(train_ds) * 0.9)
    rand_idx = torch.randperm(len(train_ds)).tolist()

    train_ds = Subset(train_ds, rand_idx[:train_len])
    valid_ds = Subset(valid_ds, rand_idx[train_len:])

    return train_ds, valid_ds


def collate_fn(batch):
    frames, flow = zip(*batch) #convert a batch of (frames, flow) to frames-batch and flow-batch
    frame1, frame2 = zip(*frames) #convert a batch of (frame1, frame 2) to frame1-batch and frame2-batch
    return (torch.stack(frame1), torch.stack(frame2)), torch.stack(flow)


def build_dl(train_ds: Subset, 
             valid_ds: Subset,
             batch_size: int,
             num_workers: int) -> Tuple[DataLoader, DataLoader]:

    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True,
                          collate_fn=None)

    valid_dl = DataLoader(valid_ds,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=False,
                          collate_fn=None)

    return train_dl, valid_dl


def build_spynets(k: int, name: str, 
                  previous: Sequence[torch.nn.Module]) -> Tuple[SpyModel.SpyNetUnit, SpyModel.SpyNet]:
    """Building a sequence of networks that contains the current trainable network and the previous layers'

    Args:
        k (int): the level of the current netwrok which we want to train
        name (str): name of the file of the pretrained models for the current level
        previous (Sequence[torch.nn.Module]): a Sequence of previous networks(if available)

    Raises:
        ValueError: if name of the pretrained is wrong

    Returns:
        Tuple[SpyModel.SpyNetUnit, SpyModel.SpyNet]: a sequence of previously trained networks with the added last network to be trained
    """


    if name != None:
        # pretrained = SpyModel.SpyNet.from_pretrained(name, map_location=device)
        # current_train = pretrained.units[k]
        raise ValueError("No saved network yet!")
    else:
        current_train = SpyModel.SpyNetUnit()
        
    current_train.to(device)
    current_train.train()
    
    if k == 0:
        Gk = None
    else:
        Gk = SpyModel.SpyNet(previous)
        Gk.to(device)
        Gk.eval()

    return current_train, Gk


def train_one_level(k: int, 
                    previous: Sequence[SpyModel.SpyNetUnit],
                    **kwargs) -> SpyModel.SpyNetUnit:
    """Train a specific level of pyramid for all epochs

    Args:
        k (int): level of the pyramid to be trained
        previous (Sequence[SpyModel.SpyNetUnit]): a list/sequence of previous trained levels

    Returns:
        SpyModel.SpyNetUnit: trained k-th level
    """

    print(f'Training level {k}...')

    train_ds, valid_ds = load_data(kwargs['root'], k)
    train_dl, valid_dl = build_dl(train_ds, valid_ds, 
                                  kwargs['batch_size'],
                                  kwargs['dl_num_workers'])

    current_level, trained_pyramid = build_spynets(k, 
                                    kwargs['finetune_name'], previous)
    optimizer = torch.optim.Adam(current_level.parameters(),
                                  lr=1e-4,
                                  weight_decay=4e-5)
    lr_sched = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.1)
    loss_fn = SpyModel.EPELoss()

    for epoch in range(kwargs['epochs']):
        print(f'---@ epoch {epoch}...')
        train_one_epoch(train_dl, 
                        optimizer,
                        loss_fn,
                        current_level,
                        trained_pyramid,
                        print_freq=100,
                        header=f'Level_{k} @ Epoch {epoch}')
        lr_sched.step()
        torch.save(current_level, str(Path(kwargs['checkpoint_dir']) / f'level{k}_e{epoch}.pt')) 
    
    return current_level


def train(**kwargs):
    torch.manual_seed(0)
    previous = []
    for k in range(kwargs.pop('levels')):
        previous.append(train_one_level(k, previous, **kwargs))

    final = SpyModel.SpyNet(previous)
    torch.save(final.state_dict(), str(Path(kwargs['checkpoint_dir']) / f'final.pt'))
    
    final.eval()
    _, valid_ds = load_data(kwargs['root'])
    frames_raw, true_flow = valid_ds[425] #random.randint(0, len(valid_ds) - 1)
    frames = [o.unsqueeze(0).to(device) for o in frames_raw]
    with torch.no_grad():
        flow_est = final(frames)[0]

    pred_of_im = utils.flow_to_image(flow_est)
    true_of = utils.flow_to_image(true_flow)

    utils.imgplot(frames_raw[0], frames_raw[1], true_of, pred_of_im)


# @click.command()

# @click.option('--root', 
#               type=click.Path(file_okay=False, exists=True))

# @click.option('--checkpoint-dir', 
#               type=click.Path(file_okay=False), default='./models/spynet')
# @click.option('--finetune-name', 
#               type=click.Choice(AVAILABLE_PRETRAINED), 
#               default='none')

# @click.option('--epochs', type=int, default=8)
# @click.option('--batch-size', type=int, default=16)
# @click.option('--dl-num-workers', type=int, default=4)

# @click.option('--levels', type=int, default=4)

def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    print(device)
    str_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model_dir = f"./models/tanh_scale_noNormalize{str_time}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir, parents=True)

    train(root='./Monkaa_cleanpass',
          checkpoint_dir=model_dir,
          finetune_name=None,
          epochs=50,
          batch_size = 16,
          dl_num_workers=4,
          levels=4)

    
