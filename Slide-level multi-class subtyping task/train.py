from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import ast
'''
path_to_feature_filepath/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
'''


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--dataset', type=str, default='', required=True,
                    choices=['BRCA', 'RCC', 'CAM16', 'PANDA', 'NSCLC'], help='dataset select')
parser.add_argument('--data_root_dir', type=str, default='path_to_feature_filepath', 
                    help='data directory')
parser.add_argument('--csv_path', type=str, default='dataset_csv/PANDA_subtyping2.csv', help='csv file')
parser.add_argument('--exp_info', type=str, default='experiment_task_2_tumor_subtyping_panda.txt', help='experiment info file')
parser.add_argument('--label_dict', default="{'grades0':0, 'grades1':1}", help='label dict')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default='PANDA_subtyping2', 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str,default='task_2_tumor_subtyping_panda', help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'cl'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str,default='task_2_tumor_subtyping', choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default='svm',
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=True, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'the device is {device}')

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
if os.path.exists(args.exp_info):
    print('load setting from file')
    with open(args.exp_info, 'r') as f:
        exp_info = f.read()
    exp_set_info = ast.literal_eval(exp_info)
    args.k = exp_set_info['num_splits']
    args.k_start = exp_set_info['k_start']
    args.k_end = exp_set_info['k_end']
    args.task = exp_set_info['task']
    args.max_epochs = exp_set_info['max_epochs']
    args.results_dir = exp_set_info['results_dir']
    args.lr = exp_set_info['lr']
    args.reg = exp_set_info['reg']
    args.label_frac = exp_set_info['label_frac']
    args.bag_loss = exp_set_info['bag_loss']
    args.seed = exp_set_info['seed']
    args.model_type = exp_set_info['model_type']
    args.model_size = exp_set_info['model_size']
    args.drop_out = exp_set_info["use_drop_out"]
    args.weighted_sample = exp_set_info['weighted_sample']
    args.opt = exp_set_info['opt']
    args.bag_weight = exp_set_info['bag_weight']
    args.inst_loss = exp_set_info['inst_loss']
    args.B = exp_set_info['B']
    args.n_classes = exp_set_info['n_classes']
    # args.split_dir = exp_set_info['split_dir']

datasetdict = {
        'BRCA':[2, {'IDC':0, 'ILC':1}],
        'RCC':[3, {'CCRCC':0, 'CHRCC':1, 'PRCC':2}],
        'NSCLC':[2, {'LUAD':0, 'LUSC':1}],
        'CAM16':[2, {'normal':0, 'tumor':1}],
        'PANDA':[2, {'grades0':0, 'grades1':1}],
    }

seed_torch(args.seed)
# encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    # args.n_classes=2
    # dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
    #                         data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
    #                         shuffle = False, 
    #                         seed = args.seed, 
    #                         print_info = True,
    #                         label_dict = {'normal_tissue':0, 'tumor_tissue':1},
    #                         patient_strat=False,
    #                         ignore=[])
    raise NotImplementedError

elif args.task == 'task_2_tumor_subtyping':
    label_dict = datasetdict[args.dataset][1]
    # label_dict = ast.literal_eval(args.label_dict)
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            # data_dir= os.path.join(args.data_root_dir, 'data_pt'),
                            data_dir= args.data_root_dir, 
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
        
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    # args.split_dir = os.path.join('splits', args.split_dir)
    args.split_dir = args.split_dir

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


