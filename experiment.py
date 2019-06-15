from __future__ import print_function

import argparse
import math
import matplotlib
import sys

import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb-clusters', required = True, type = int)
    parser.add_argument('--dataset', dest = 'dataset_selected',
        choices=['sop', 'inshop', 'vid'], required = True
    )
    parser.add_argument('--nb-epochs', type = int, default=200)
    parser.add_argument('--finetune-epoch', type = int, default=190)
    parser.add_argument('--mod-epoch', type = int, default=2)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--sz-batch', type=int, default=80)
    parser.add_argument('--sz-embedding', default=128, type=int)
    parser.add_argument('--cuda-device', default = 0, type = int)
    parser.add_argument('--exp', default='0', type=str, help='experiment identifier')
    parser.add_argument('--dir', default='default', type=str)
    parser.add_argument('--backend', default='faiss',
        choices=('torch+sklearn', 'faiss', 'faiss-gpu')
    )
    parser.add_argument('--random-seed', default = 0, type = int)
    parser.add_argument('--backbone-wd', default=1e-4, type=float)
    parser.add_argument('--backbone-lr', default=1e-5, type=float)
    parser.add_argument('--embedding-lr', default=1e-5, type=float)
    parser.add_argument('--embedding-wd', default=1e-4, type=float)
    parser.add_argument('--verbose', action = 'store_true')
    args = vars(parser.parse_args())

    config = train.load_config(config_name = 'config.json')

    config['dataloader']['batch_size'] = args.pop('sz_batch')
    config['dataloader']['num_workers'] = args.pop('num_workers')
    config['recluster']['mod_epoch'] = args.pop('mod_epoch')
    config['opt']['backbone']['lr'] = args.pop('backbone_lr')
    config['opt']['backbone']['weight_decay'] = args.pop('backbone_wd')
    config['opt']['embedding']['lr'] = args.pop('embedding_lr')
    config['opt']['embedding']['weight_decay'] = args.pop('embedding_wd')

    for k in args:
        if k in config:
            config[k] = args[k]

    if config['nb_clusters'] == 1:
        config['recluster']['enabled'] = False

    config['log'] = {
        'name': '{}-K-{}-M-{}-exp-{}'.format(
            config['dataset_selected'],
            config['nb_clusters'],
            config['recluster']['mod_epoch'],
            args['exp']
        ),
        'path': 'log/{}'.format(args['dir'])
    }

    # tkinter not installed on this system, use non-GUI backend
    matplotlib.use('agg')
    train.start(config)

