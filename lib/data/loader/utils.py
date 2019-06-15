
from __future__ import print_function
from __future__ import division

import random

import torch
import numpy as np
from ..set import VehicleID, InShop, SOProducts
from ..set import transform
from .sampler import ClassBalancedSampler


datasets = {
    'sop': SOProducts,
    'inshop': InShop,
    'vid': VehicleID
}


def make(config, model, type, subset_indices = None, inshop_type = None):
    """
    subset_indices: indices for selecting subset of dataset, for creating
        clustered dataloaders.
    type: 'init', 'eval' or 'train'.
    """
    # inshop_types: train, query, gallery; basically instead of labels/classes
    ds_name = config['dataset_selected']
    if ds_name == 'inshop':
        ds = datasets[ds_name](
            root = config['dataset'][ds_name]['root'],
            dset_type = inshop_type,
            transform = transform.make(
                **config['transform_parameters'],
                is_train = True if type == 'train' else False
            )
        )
    else:
        ds = datasets[ds_name](
            root = config['dataset'][ds_name]['root'],
            classes = config['dataset'][ds_name]['classes'][type],
            transform = transform.make(
                **config['transform_parameters'],
                is_train = True if type == 'train' else False
            )
        )
    if type == 'train':
        ds.set_subset(subset_indices)
        _c = config['dataloader']
        dl = torch.utils.data.DataLoader(
            ds,
            # ignore batch_size, since batch_sampler enabled
            **{k: _c[k] for k in _c if k != 'batch_size'},
            batch_size = -1,
            batch_sampler = ClassBalancedSampler(
                ds,
                batch_size = config['dataloader']['batch_size'],
                num_samples_per_class = 4
            )
        )
    else:
        # else init or eval loader
        dl = torch.utils.data.DataLoader(ds, **config['dataloader'])
    return dl


def make_from_clusters(C, subset_indices, model, config):
    import numpy as np
    from math import ceil
    dataloaders = [[None] for c in range(config['nb_clusters'])]
    for c in range(config['nb_clusters']):
        dataloaders[c] = make(
            config = config, model = model, type = 'train', subset_indices = subset_indices[C == c],
            inshop_type = 'train')
        dataloaders[c].dataset.id = c
    return dataloaders


def merge(dls_non_iter):

    nb_batches_per_dl = [len(dl) for dl in dls_non_iter]
    nb_batches = max(nb_batches_per_dl)
    I = range(len(dls_non_iter))
    length = len(dls_non_iter)
    dls = [iter(dl) for dl in dls_non_iter]

    for j in range(nb_batches):
        for i in I:
            b = next(dls[i], None)
            if b == None:
                # initialize new dataloader in case no batches left
                dls[i] = iter(dls_non_iter[i])
                b = next(dls[i])
            yield b, dls[i].dataset

