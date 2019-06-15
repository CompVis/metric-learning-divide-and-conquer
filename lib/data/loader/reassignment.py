
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np


def reassign_clusters(C_prev, C_curr, I_prev, I_curr):
    nb_clusters = max(C_prev).item() + 1 # cluster ids start from 0
    assert set(
        i.item() for i in np.unique(I_prev)
    ) == set(i.item() for i in np.unique(I_curr))
    I_max = max(I_curr).item() + 1
    I_all = {
        'prev': torch.zeros(nb_clusters, I_max),
        'curr': torch.zeros(nb_clusters, I_max)
    }
    I = {'prev': I_prev, 'curr': I_curr}
    C = {'prev': C_prev, 'curr': C_curr}

    for e in ['prev', 'curr']:
        for c in range(nb_clusters):
            _C = C[e]
            _I = I[e]
            I_all[e][c, _I[_C == c]] = 1

    costs = torch.zeros(nb_clusters, nb_clusters)
    for i in range(nb_clusters):
        for j in range(nb_clusters):
            costs[i, j] = torch.norm(
                I_all['curr'][i] - I_all['prev'][j],
                p = 1
            )

    reassign_prev, reassign_curr = linear_sum_assignment(costs)

    C_reassigned = C['curr'].copy()

    for a_prev, a_curr in zip(reassign_prev, reassign_curr):
        C_reassigned[C['curr'] == int(a_curr)] = int(a_prev)

    return C_reassigned, costs
