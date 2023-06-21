import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from datasets import SubDataset
from tqdm import tqdm

from genmodel import genmodel 




def get_args():
    parser = argparse.ArgumentParser('extract features script for whole slide image classification', add_help=False)
    parser.add_argument('--sub_batch_size', default=128, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_root_path', type=str, default='', required=True)
    parser.add_argument('--save_pt_path', type=str, default='', required=True)
    parser.add_argument('--pin_mem', action='store_true', default=True, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--modelpath', default='', required=True, help='Dir to the checkpoints')
    parser.add_argument('--dataset', choices=['BRCA', 'RCC', 'CAM16', 'PANDA', 'NSCLC'], required=True,  help='select the dataset')
    parser.add_argument('--file_ext', type=str, default='.svs', help='setting the file extension')


    return parser.parse_args()

 
def extract_features(args):
    device = torch.device(args.device)
    model = genmodel(ckpt=args.modelpath)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    csv_path = f'./csv_files/{args.dataset}.csv'

    df = pd.read_csv(csv_path)
    wsidirs = df['slide_id'].to_list()
    for i in range(len(wsidirs)):
        print(f'working on {i} / {len(wsidirs)}')
        wsidir = f'{args.data_root_path}/{wsidirs[i]}' + args.file_ext
        sub_dataset = SubDataset(wsidir)

        # 创建子数据加载器
        sub_dataloader = DataLoader(sub_dataset, batch_size=args.sub_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        name = wsidir.split('/')[-1]
        ext = name.split('.')[-1]
        lenext = len(ext) + 1 
        name = name[:-lenext]
        savedir = f'{args.save_pt_path}/{name}.pt'
        if os.path.exists(savedir):
            continue
        if True:
            out = []
            for i, patch in enumerate(tqdm(sub_dataloader)):
                patch = patch.to(device)

                with torch.no_grad():
                    output = model(patch)
                    features = output.cpu().detach()
                    # print(features.shape)
                out.append(features)
            out = torch.cat(out, dim=0)
            torch.save(out, savedir)


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_pt_path, exist_ok=True)
    extract_features(args)
