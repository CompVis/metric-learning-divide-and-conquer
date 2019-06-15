from __future__ import print_function
from __future__ import division

import numpy as np
import faiss
import sys
import time
import warnings
import logging


if not sys.warnoptions:
    # suppress pesky PIL EXIF warnings
    warnings.simplefilter("once")
    warnings.filterwarnings("ignore", message="(Possibly )?corrupt EXIF data.*")
    warnings.filterwarnings("ignore", message="numpy.dtype size changed.*")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed.*")


def reserve_faiss_gpu_memory(gpu_id=0):
    """
    Reserves around 2.4 Gb memory on Titan Xp.
    `r = reserve_faiss_gpu_memory()`
    To release the memory run `del r`

    Something like 200 Mb will still be hold afterwards.
    """
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id
    index = faiss.GpuIndexFlatL2(res, 2048, cfg)
    return index, res


class MemoryReserver():
    """
    Faiss memory manager. If not used and another process takes up memory of
    currently used GPU, then the program will crash.
    """
    def __init__(self):
        self.memory_holder = None

    def lock(self, backend):
        # reserve memory for faiss if backend is faiss-gpu
        if backend == 'faiss-gpu':
            logging.debug('Reserve some memory for FAISS')
            self.memory_holder = reserve_faiss_gpu_memory(gpu_id=0)
        else:
            self.memory_holder = None

    def release(self):
        if self.memory_holder is not None:
            logging.debug('Release memory for FAISS')
            self.memory_holder = None


def preprocess_features(x, x2=None, d=256):
    """
    Calculate PCA + Whitening + L2 normalization for each vector

    Args:
        x (ndarray): N x D, where N is number of vectors, D - dimensionality
        x2 (ndarray): optional, if not None apply PCA+Whitening learned on x to x2.
        d (int): number of output dimensions (how many principal components to use).
    Returns:
        transformed [N x d] matrix xt .
    """
    n, orig_d = x.shape
    pcaw = faiss.PCAMatrix(d_in=orig_d, d_out=d, eigen_power=-0.5, random_rotation=False)
    pcaw.train(x)
    assert pcaw.is_trained
    print('Performing PCA + whitening')
    x = pcaw.apply_py(x)
    print('x.shape after PCA + whitening:', x.shape)
    l2normalization = faiss.NormalizationTransform(d, 2.0)
    print('Performing L2 normalization')
    x = l2normalization.apply_py(x)
    if x2 is not None:
        print('Perform PCA + whitening for x2')
        x2 = pcaw.apply_py(x2)
        x2 = l2normalization.apply_py(x2)
        return x, x2
    else:
        return x


def train_kmeans(x, num_clusters=1000, gpu_ids=None, niter=100, nredo=1, verbose=0):
    """
    Runs k-means clustering on one or several GPUs
    """
    assert np.all(~np.isnan(x)), 'x contains NaN'
    assert np.all(np.isfinite(x)), 'x contains Inf'
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    assert gpu_ids is None or len(gpu_ids)

    d = x.shape[1]
    kmeans = faiss.Clustering(d, num_clusters)
    kmeans.verbose = bool(verbose)
    kmeans.niter = niter
    kmeans.nredo = nredo

    # otherwise the kmeans implementation sub-samples the training set
    kmeans.max_points_per_centroid = 10000000

    if gpu_ids is not None:
        res = [faiss.StandardGpuResources() for i in gpu_ids]

        flat_config = []
        for i in gpu_ids:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = i
            flat_config.append(cfg)

        if len(gpu_ids) == 1:
            index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
        else:
            indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                       for i in range(len(gpu_ids))]
            index = faiss.IndexProxy()
            for sub_index in indexes:
                index.addIndex(sub_index)
    else:
        index = faiss.IndexFlatL2(d)

    # perform the training
    kmeans.train(x, index)
    centroids = faiss.vector_float_to_array(kmeans.centroids)

    objective = faiss.vector_float_to_array(kmeans.obj)
    #logging.debug("Final objective: %.4g" % objective[-1])

    return centroids.reshape(num_clusters, d)


def compute_cluster_assignment(centroids, x):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel()


def do_clustering(features, num_clusters, gpu_ids=None,
                  num_pca_components=None, niter=100, nredo=1, verbose=0):
    logging.debug('FAISS: using GPUs {}'.format(gpu_ids))
    features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)

    if num_pca_components is not None:
        features = preprocess_features(features, d=num_pca_components,
                                       niter=niter, nredo=nredo, verbose=verbose)

    logging.debug('FAISS: clustering...')
    t0 = time.time()
    centroids = train_kmeans(features, num_clusters, gpu_ids=gpu_ids, verbose=1)
    labels = compute_cluster_assignment(centroids, features)
    t1 = time.time()
    logging.debug("FAISS: Clustering total elapsed time: %.3f m" % ((t1 - t0) / 60.0))
    return labels


def find_nearest_neighbors(x, queries=None, k=5, gpu_id=None):
    """
    Find k nearest neighbors for each of the n examples.
    Distances are computed using Squared Euclidean distance metric.

    Arguments:
    ----------
    queries
    x (ndarray): N examples to search within. [N x d].
    gpu_id (int): use CPU if None else use GPU with the specified id.
    queries (ndarray): find nearest neigbor for each query example. [M x d] matrix
        If None than find k nearest neighbors for each row of x
        (excluding self exampels).
    k (int): number of nearest neighbors to find.

    Return
    I (ndarray): Indices of the nearest neighnpors. [M x k]
    distances (ndarray): Distances to the nearest neighbors. [M x k]

    """
    if gpu_id is not None and not isinstance(gpu_id, int):
        raise ValueError('gpu_id must be None or int')
    x = np.asarray(x.reshape(x.shape[0], -1), dtype=np.float32)
    remove_self = False # will we have queries in the search results?
    if queries is None:
        remove_self = True
        queries = x
        k += 1

    d = x.shape[1]

    tic = time.time()
    if gpu_id is None:
        logging.debug('FAISS: cpu::find {} nearest neighbors'\
                     .format(k - int(remove_self)))
        index = faiss.IndexFlatL2(d)
    else:
        logging.debug('FAISS: gpu[{}]::find {} nearest neighbors'\
                     .format(gpu_id, k - int(remove_self)))
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id

        flat_config = [cfg]
        resources = [faiss.StandardGpuResources()]
        index = faiss.GpuIndexFlatL2(resources[0], d, flat_config[0])
    index.add(x)
    distances, nns = index.search(queries, k)
    if remove_self:
        for i in range(len(nns)):
            indices = np.nonzero(nns[i, :] != i)[0]
            indices.sort()
            if len(indices) > k - 1:
                indices = indices[:-1]
            nns[i, :-1] = nns[i, indices]
            distances[i, :-1] = distances[i, indices]
        nns = nns[:, :-1]
        distances = distances[:, :-1]
    logging.debug('FAISS: Neighbors search total elapsed time: {:.2f} sec'.format(time.time() - tic))
    return nns, distances


def example(size=30000, k=10, num_pca_components=256):
    gpu_ids = [0]

    x = np.random.rand(size, 512)
    print("reshape")
    x = x.reshape(x.shape[0], -1).astype('float32')
    x, _ = preprocess_features(x, x, d=num_pca_components)

    print("run")
    t0 = time.time()
    centroids = train_kmeans(x, k, gpu_ids=gpu_ids)
    print('compute_cluster_assignment')
    labels = compute_cluster_assignment(centroids, x)
    print('centroids.shape:', centroids.shape)
    print('labels.type:', labels.__class__, labels.dtype)
    print('labels.shape:', labels.shape)
    t1 = time.time()

    print("total runtime: %.2f s" % (t1 - t0))


def test_knn_search(size=10000, gpu_id=None):
    x = np.random.rand(size, 512)
    x = x.reshape(x.shape[0], -1).astype('float32')
    d = x.shape[1]

    tic = time.time()
    if gpu_id is None:
        index = faiss.IndexFlatL2(d)
    else:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id

        flat_config = [cfg]
        resources = [faiss.StandardGpuResources()]
        index = faiss.GpuIndexFlatL2(resources[0], d, flat_config[0])
    index.add(x)
    print('Index built in {} sec'.format(time.time() - tic))
    distances, I = index.search(x, 21)
    print('Searched in {} sec'.format(time.time() - tic))
    print(distances.shape)
    print(I.shape)
    print(distances[:5])
    print(I[:5])


if __name__ == '__main__':
    #example(size=100000, k=3, num_pca_components=32)
    test_knn_search(size=100000, gpu_id=5)
