from __future__ import print_function
from __future__ import division

import logging
import random
import torch
import sklearn
import time
import numpy as np

from . import utils
from . import faissext

__all__ = [
    'pairwise_distance',
    'assign_by_euclidian_at_k'
]


TORCH_SKLEARN_BACKEND = 'torch+sklearn'
FAISS_BACKEND = 'faiss'
FAISS_GPU_BACKEND = 'faiss-gpu'
_DEFAULT_BACKEND_ = FAISS_GPU_BACKEND
_backends_ = [TORCH_SKLEARN_BACKEND, FAISS_BACKEND, FAISS_GPU_BACKEND]


def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.
    Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    a = torch.as_tensor(np.atleast_2d(a))
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals).data.cpu().numpy()

    return pairwise_distances


def assign_by_euclidian_at_k(X, T, k, gpu_id=None, backend=_DEFAULT_BACKEND_):
    """
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    if backend == TORCH_SKLEARN_BACKEND:
        distances = sklearn.metrics.pairwise.pairwise_distances(X)
        # get nearest points
        nns = np.argsort(distances, axis = 1)[:, :k + 1]
        for i in range(len(nns)):
            indices = np.nonzero(nns[i, :] != i)[0]
            if len(indices) > k:
                indices = indices[:-1]
            nns[i, :-1] = nns[i, indices]
        nns = nns[:, :-1]
        assert nns.shape[1] == k, nns.shape
    else:
        nns, _ = faissext.find_nearest_neighbors(X,
                                                 k=k,
                                                 gpu_id=None if backend != FAISS_GPU_BACKEND
                                                    else torch.cuda.current_device()
        )
    return np.array([[T[i] for i in ii] for ii in nns])


def cluster_by_kmeans(X, nb_clusters, gpu_id=None, backend=_DEFAULT_BACKEND_):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    if backend == TORCH_SKLEARN_BACKEND:
        C = sklearn.cluster.KMeans(nb_clusters).fit(X).labels_
    else:
        C = faissext.do_clustering(
            X,
            num_clusters = nb_clusters,
            gpu_ids = None if backend != FAISS_GPU_BACKEND
                else torch.cuda.current_device(),
            niter=100,
            nredo=5,
            verbose=1
        )
    return C

