import argparse
import os
import sys
import yaml
import glob
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import seaborn as sns

from copy import deepcopy
from importlib import reload
from time import sleep
from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.datasets import CelebA, VisionDataset

import torchmetrics

from attgan import AttGAN
from utils import find_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('attr_npy_path', type=str)
    parser.add_argument('x2y_ckpt', type=str)
    parser.add_argument('--gen_dset_dirname', type=str, default=None, 
                        help="Usually something related to attr_npy_path.")
    parser.add_argument('-k', type=int, default=1, help="1, 2, and 3 are valid choices for k.")
    parser.add_argument('--bsize', type=int, default=256)
    parser.add_argument('--data_root_dir', type=str, default='/home/andrewbai/data/')
    parser.add_argument('--attr_root_dir', type=str, default='/home/andrewbai/attrs/')
    parser.add_argument('--attgan_ckpt_dir', type=str, default='/nfs/data/andrewbai/AttGAN-PyTorch/output')
    
    return parser.parse_args()
    
class ClasslessVisionDataset(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.img_fnames = sorted(glob.glob(f"{root}/*.jpg"))
        assert len(self.img_fnames) > 0
        
    def __len__(self):
        return len(self.img_fnames)
    
    def __getitem__(self, idx):
        image = PIL.Image.open(self.img_fnames[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    
def main():
    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiment_name = '256_shortcut1_inject1_none'
    attgan_img_size = 256
    img_size = 299
    img_start_index = 162771
    
    # check if generate new images cache exist and load it. (force option)
    if args.gen_dset_dirname is None:
        args.gen_dset_dirname = args.attr_npy_path.split('.')[0]
    args.gen_dset_dirname += f'_k{args.k}'
    celeba_gen_dir = os.path.join(args.data_root_dir, 'celeba_attgan', args.gen_dset_dirname)
    
    if not os.path.exists(celeba_gen_dir):
        
        # load AttGAN model.

        with open(os.path.join(args.attgan_ckpt_dir, experiment_name, 'setting.txt'), 'r') as f:
            attgan_args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        attgan_args.gpu = True
        attgan_args.multi_gpu = False

        attgan = AttGAN(attgan_args)
        attgan.load(find_model(os.path.join(args.attgan_ckpt_dir, experiment_name, 'checkpoint'), 'latest'))
        attgan.eval()
        attgan.G.to(device)

        # load dataset (CelebA valid).

        transform = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(attgan_img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dset = CelebA(root=args.data_root_dir, split='valid', 
                      transform=transform, target_type="attr")
        dl = DataLoader(dset, batch_size=args.bsize, shuffle=False,
                        drop_last=False, num_workers=8)
        
        all_attrs = np.array("5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split(' '))
        attr_mask = np.isin(all_attrs, attgan_args.attrs)

        # load attribution from npy.
        concept_attributions = np.load(os.path.join(args.attr_root_dir, args.attr_npy_path))

        # iterate over dataset and generate new images with modified top k attributes.
        gen_xs, filtered_clipped_cs = [], []
        for i, (xs, cs) in enumerate(tqdm(dl)):
            # filter cs and attributions with attgan_attr_mask
            xs = xs.to(device)
            cs = cs[:, attr_mask]
            cas = torch.from_numpy(concept_attributions[i * args.bsize:i * args.bsize + cs.shape[0], attr_mask]).bool()
            
            # get cs_flip_mask according to attribution * cs
            # cas positive needs to match with cs negative
            cs_flip_mask = ((cs - 0.5) * cas).argsort(-1) < args.k

            # flip cs
            cs[cs_flip_mask] = 1 - cs[cs_flip_mask]
            filtered_clipped_cs.append(cs.detach().cpu().numpy())
            
            cs = (cs * 2 - 1) * attgan_args.thres_int
            cs = cs.to(device)

            # generate
            with torch.no_grad():
                gen_xs_ = (attgan.G(xs, cs) + 1.) / 2.
                gen_xs_ = gen_xs_.permute(0, 2, 3, 1)
                gen_xs.append(gen_xs_.detach().cpu().numpy())
        gen_xs = np.concatenate(gen_xs, axis=0)
        filtered_clipped_cs = np.concatenate(filtered_clipped_cs, axis=0)
    
        # save new images in cache.
        os.makedirs(os.path.join(celeba_gen_dir, 'images'))
        
        for i, gen_x in enumerate(tqdm(gen_xs, leave=False)):
            im = PIL.Image.fromarray((gen_x * 255).astype(np.uint8))
            im.save(os.path.join(celeba_gen_dir, f"images/{img_start_index + i:06}.jpg"))
        
        np.save(os.path.join(celeba_gen_dir, 'filtered_clipped_cs.npy'), filtered_clipped_cs)
        
        del dset, dl, attgan
    
    # optional: check whether new images classified with flipped attribute with x2c.
    
    # load x2y model.
    model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
    model.AuxLogits.fc = nn.Linear(768, 1)
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(args.x2y_ckpt))
    model = model.eval().to(device)
    
    # iterate original and new generated images (different img_size), evaluate performance decrease in x2y task.
    transform = transforms.Compose([
        transforms.CenterCrop(170),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dset_orig = CelebA(root=args.data_root_dir, split='valid', 
                       transform=transform, target_type="attr")
    dset_gen = ClasslessVisionDataset(os.path.join(celeba_gen_dir, 'images'), transform=transform)
    dl_orig = DataLoader(dset_orig, batch_size=args.bsize, shuffle=False,
                         drop_last=False, num_workers=8)
    dl_gen = DataLoader(dset_gen, batch_size=args.bsize, shuffle=False,
                        drop_last=False, num_workers=8)
    
    # print results.
    pred_diffs = []
    for (xs, cs), gen_xs in tqdm(zip(dl_orig, dl_gen), total=len(dl_orig)):
        xs = xs.to(device)
        gen_xs = gen_xs.to(device)
        
        with torch.no_grad():
            pred_diff = model(xs) - model(gen_xs)
            pred_diffs.append(pred_diff.detach().cpu().numpy())
            
    pred_diffs = np.concatenate(pred_diffs, axis=0)
    print(f"average score diff: {pred_diffs.mean()}")
            
if __name__ == '__main__':
    main()