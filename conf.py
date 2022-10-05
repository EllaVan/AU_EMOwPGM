import datetime
import argparse
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import os
import argparse
import pprint
import numpy as np
import random
import logging
import shutil
import yaml

import torch
import models

parser = argparse.ArgumentParser(description='PyTorch Training')
# Datasets
parser.add_argument('--dataset', default="RAF-DB-compound", type=str, help="experiment dataset BP4D / DISFA")
parser.add_argument('--N-fold', default=3, type=int, help="the ratio of train and validation data")
parser.add_argument('-f','--fold', default=0, type=int, metavar='N', help='the fold of three folds cross-validation ')

# overall experiments setting
parser.add_argument('-b','--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-start_epoch', default=0, type=int, help='start_epoch')
parser.add_argument('-e', '--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-save_epoch', default=4, type=int, help='save_epoch')
parser.add_argument('--exp-name', default="Test", type=str, help="experiment name for saving checkpoints")
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')

# Device and Seed
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')

# GraphAU setting
parser.add_argument('-lr_AU', '--learning-rate_AU', default=0.0001, type=float, metavar='LR', help='initial learning rate for AU')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer-eps', default=1e-8, type=float)
parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
parser.add_argument('--evaluate', action='store_true', help='evaluation mode')
parser.add_argument('--arc', default='resnet18', type=str, choices=['resnet18', 'resnet50', 'resnet101',
                    'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base'], help="backbone architecture resnet / swin_transformer")
parser.add_argument('--metric', default="dots", type=str, choices=['dots', 'cosine', 'l1'], help="metric for graph top-K nearest neighbors selection")
parser.add_argument('--lam_AU', default=0.001, type=float, help="lambda for adjusting loss")

# EAC
parser.add_argument('-lr_EMO', '--learning-rate_EMO', default=0.0003, type=float, metavar='LR', help='initial learning rate for EMO')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--lam_EMO', type=float, default=5, help='kl_lambda')

# Rules
parser.add_argument('--lr_relation', type=float, default=0.001)
parser.add_argument('--lr_decay_idx', type=int, default=20000)
parser.add_argument('--AUthresh', type=float, default=0.6)
parser.add_argument('--zeroPad', type=float, default=1e-4)

#SSL
parser.add_argument('--SSL_temperature', type=float, default=0.5)
parser.add_argument('--SSL_nodes', type=int, default=1)

# StableLearning
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser.add_argument('--lr_stable', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--cos', '--cosine_lr', default=1, type=int,
                    metavar='COS', help='lr decay by decay', dest='cos')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
                    
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# for number of fourier spaces
parser.add_argument ('--num_f', type=int, default=1, help = 'number of fourier spaces')

parser.add_argument ('--sample_rate', type=float, default=1.0, help = 'sample ratio of the features involved in balancing')
parser.add_argument ('--lrbl', type = float, default = 1.0, help = 'learning rate of balance')

# parser.add_argument ('--cfs', type = int, default = 512, help = 'the dim of each feature')
parser.add_argument ('--lambdap', type = float, default = 70.0, help = 'weight decay for weight1 ')
parser.add_argument ('--lambdapre', type = float, default = 1, help = 'weight for pre_weight1 ')

parser.add_argument ('--epochb', type = int, default = 20, help = 'number of epochs to balance')
parser.add_argument ('--epochp', type = int, default = 0, help = 'number of epochs to pretrain')

parser.add_argument ('--n_feature', type=int, default=128, help = 'number of pre-saved features')
parser.add_argument ('--feature_dim', type=int, default=512, help = 'the dim of each feature')

parser.add_argument ('--lrwarmup_epo', type=int, default=0, help = 'the dim of each feature')
parser.add_argument ('--lrwarmup_decay', type=int, default=0.1, help = 'the dim of each feature')

parser.add_argument ('--n_levels', type=int, default=1, help = 'number of global table levels')

# for expectation
parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')

# for jointly train
parser.add_argument ('--train_cnn_with_lossb', type=bool, default=False, help = 'whether train cnn with lossb')
parser.add_argument ('--cnn_lossb_lambda', type=float, default=0, help = 'lambda for lossb')

# for more moments
parser.add_argument ('--moments_lossb', type=float, default=1, help = 'number of moments')

# for first step
parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

# for pow
parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')

# for second order moment weight
parser.add_argument ('--second_lambda', type=float, default=0.2, help = 'weight lambda for second order moment loss')
parser.add_argument ('--third_lambda', type=float, default=0.05, help = 'weight lambda for second order moment loss')

# for dat/.a aug
parser.add_argument ('--lower_scale', type=float, default=0.8, help = 'weight lambda for second order moment loss')

# for lr decay epochs
parser.add_argument ('--epochs_decay', type=list, default=[24, 30], help = 'weight lambda for second order moment loss')

parser.add_argument ('--classes_num', type=int, default=5, help = 'number of epoch for lambda to decay')

parser.add_argument ('--sub_dataset', type=str, default="", help = '')
parser.add_argument ('--gray_scale', type=float, default=0.1, help = 'weight lambda for second order moment loss')

parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--concat', type=int, default=1, help='sum or concat')
parser.add_argument('--min_scale', type=float, default=0.8, help='')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')
# ------------------------------


def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)


def str2bool(v):
    return v.lower() in ('true', '1')


# def add_argument_group(name):
def add_argument_group(arg_lists, name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ------------------------------

def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def get_config(cfg):

    # args from argparser
    # cfg = parser2dict()
    if cfg.dataset == 'BP4D':
        with open('config/BP4D_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
    elif cfg.dataset == 'DISFA':
        with open('config/DISFA_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
    elif cfg.dataset == 'RAF-DB':
        with open('config/RAF_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
    elif cfg.dataset == 'RAF-DB-compound':
        with open('config/RAF_compound_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
    elif cfg.dataset == 'AffectNet':
        with open('config/AffectNet_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
    elif cfg.dataset == 'CASME':
        with open('config/CASME_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    if cfg.fold == 0:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_independent
    else:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_dependent
    cfg.update(datasets_cfg)
    return cfg


def set_env(cfg):
    # set seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    # os.environ["NUMEXPR_MAX_THREADS"] = '16'
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids


def set_outdir(conf):
    default_outdir = 'results'
    if 'timedir' in conf:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(default_outdir,conf.exp_name,timestr)
    else:
        outdir = os.path.join(default_outdir, conf.dataset, conf.exp_name)
        if conf.fold == 0:
            exp_type = 'subject_independent'
        else:
            exp_type = 'subject_dependent'
        prefix = 'bs_'+str(conf.batch_size)+'_seed_'+str(conf.seed)+'_lrEMO_'+str(conf.learning_rate_EMO)+'_lrAU_'+str(conf.learning_rate_AU)+'_lr_relation_'+str(conf.lr_relation)
        outdir = os.path.join(outdir, exp_type, prefix)
    ensure_dir(outdir, conf.start_epoch)
    conf['outdir'] = outdir
    shutil.copyfile("./models/TwoBranch.py", os.path.join(outdir,'TwoBranch.py'))
    return conf


# check if dir exist, if not create new folder
def ensure_dir(dir_name, start_epoch):
    if start_epoch == 0:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            # print('{} is created'.format(dir_name))
        else:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            # print('The original {} is deleted, a new path is created'.format(dir_name))


def set_logger(cfg):

    if 'loglevel' in cfg:
        loglevel = eval('logging.'+loglevel)
    else:
        loglevel = logging.INFO


    if cfg.evaluate:
        outname = 'test.log'
    else:
        outname = 'train.log'

    outdir = cfg['outdir']
    log_path = os.path.join(outdir,outname)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info(print_conf(cfg))
    logging.info('writting logs to file {}'.format(log_path))
