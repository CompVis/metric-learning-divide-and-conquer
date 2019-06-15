from __future__ import print_function
from __future__ import division

import torch
import logging
import numpy as np
import sklearn.cluster
from . import evaluation
from . import faissext
from . import data
from . import utils

def get_cluster_labels(model, data_loader, use_penultimate, nb_clusters,
       gpu_id=None, backend='faiss'):
    is_dry_run = (nb_clusters == 1)
    if not is_dry_run:
        if not use_penultimate:
            logging.debug('Using the final layer for clustering')
        X_all, T_all, I_all = utils.predict_batchwise(
            model=model,
            dataloader=data_loader,
            use_penultimate=use_penultimate,
            is_dry_run=is_dry_run
        )
        perm = np.argsort(I_all)
        X_all = X_all[perm]
        I_all = I_all[perm]
        T_all = T_all[perm]
        if backend == 'torch+sklearn':
            clustering_algorithm = sklearn.cluster.KMeans(
                n_clusters=nb_clusters)
            C = clustering_algorithm.fit(X_all).labels_
        else:
            C = faissext.do_clustering(
                X_all,
                num_clusters = nb_clusters,
                gpu_ids = None if backend != 'faiss-gpu'
                    else torch.cuda.current_device(),
                niter=100,
                nredo=5,
                verbose=0
            )
    else:
        T_all = np.array(data_loader.dataset.ys)
        I_all = np.array(data_loader.dataset.I)
        C = np.zeros(len(T_all), dtype=int)
    return C, T_all, I_all


def make_clustered_dataloaders(model, dataloader_init, config,
        reassign = False, I_prev = None, C_prev = None, logging = None):

    def correct_indices(I):
        return torch.sort(torch.LongTensor(I))[1]

    C, T, I = get_cluster_labels(
        model,
        dataloader_init,
        use_penultimate = True,
        nb_clusters = config['nb_clusters'],
        backend=config['backend']
    )

    if reassign == True:

        # get correct indices for samples by sorting them and return arg sort
        I_correct = correct_indices(I)
        I = I[I_correct]
        T = T[I_correct]
        C = C[I_correct]

        # also use the same indices of sorted samples for previous data
        I_prev_correct = correct_indices(I_prev)
        I_prev = I_prev[I_prev_correct]
        C_prev = C_prev[I_prev_correct]

        logging.debug('Reassigning clusters...')
        logging.debug('Calculating NMI for consecutive cluster assignments...')
        logging.debug(str(
            evaluation.calc_normalized_mutual_information(
            C[I],
            C_prev[I_prev]
        )))

        # assign s.t. least costs w.r.t. L1 norm
        C, costs = data.loader.reassign_clusters(C_prev = C_prev,
                C_curr = C, I_prev = I_prev, I_curr = I)
        logging.debug('Costs before reassignment')
        logging.debug(str(costs))
        _, costs = data.loader.reassign_clusters(C_prev = C_prev,
                C_curr = C, I_prev = I_prev, I_curr = I)
        # after printing out the costs now, the trace of matrix should
        # have lower numbers than other entries in matrix
        logging.debug('Costs after reassignment')
        logging.debug(str(costs))

    #  remove labels s.t. minimum 2 samples per class per cluster
    for c in range(config['nb_clusters']):
        for t in np.unique(T[C == c]):
            if (T[C == c] == t).sum().item() == 1:
                # assign to cluster -1 if only one sample from class
                C[(T == t) & (C == c)] = -1

    dls = data.loader.make_from_clusters(
        C = C, subset_indices = I, model = model, config = config
    )

    return dls, C, T, I
