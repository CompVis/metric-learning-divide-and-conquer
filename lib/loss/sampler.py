
from __future__ import print_function
from __future__ import division


import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


def topk_mask(input, dim, K=10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim=dim, **kwargs)[1]
    return torch.zeros_like(input.data).scatter(dim, index, 1.0)


class Sampler(nn.Module):
    """
    Sample for each anchor negative examples
        are K closest points on the distance >= cutoff

    Inputs:
        - **data**: input tensor with shape (batch_size, embed_dim).
        Here we assume the consecutive batch_k examples are of the same class.
        For example, if batch_k = 5, the first 5 examples belong to the same class,
        6th-10th examples belong to another class, etc.

    Outputs:
        - a_indices: indices of anchors.
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        - x: embeddings of the input batch.
    """

    def __init__(self, cutoff=0.5, infinity=1e6, eps=1e-6):
        super(Sampler, self).__init__()
        self.cutoff = cutoff
        self.infinity = infinity
        self.eps = eps

    def forward(self, x, labels):
        """
        x: input tensor of shape (batch_size, embed_dim)
        labels: tensor of class labels of shape (batch_size,)
        """
        d = pdist(x)
        pos = torch.eq(
            *[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]
        ).type_as(d) - (torch.eye( len(d))).type_as(d)
        num_neg = int(pos.data.sum()) // len(pos)
        neg = topk_mask(
            d + self.infinity * ((pos > 0) + (d < self.cutoff)).type_as(d),
            dim=1,
            largest=False,
            K=num_neg
        )

        a_indices = []
        p_indices = []
        n_indices = []

        for i in range(len(d)):
            a_indices.extend([i] * num_neg)
            p_indices.extend(
                np.atleast_1d(pos[i].nonzero().squeeze().cpu().numpy())
            )
            n_indices.extend(
                np.atleast_1d(neg[i].nonzero().squeeze().cpu().numpy())
            )

            if len(a_indices) != len(p_indices) or len(a_indices) != len(n_indices):
                logging.warning(
                    'Probably too many positives, because of lacking classes in' +
                    ' the current cluster.' +
                    ' n_anchors={}, n_pos={}, n_neg= {}'.format(
                        *map(len, [a_indices, p_indices, n_indices])
                    )
                )
                min_len = min(map(len, [a_indices, p_indices, n_indices]))
                a_indices = a_indices[:min_len]
                p_indices = p_indices[:min_len]
                n_indices = n_indices[:min_len]

        assert len(a_indices) == len(p_indices) == len(n_indices), \
                '{}, {}, {}'.format(
                    *map(len, [a_indices, p_indices, n_indices])
                )

        return a_indices, x[a_indices], x[p_indices], x[n_indices]
