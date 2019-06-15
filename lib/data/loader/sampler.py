from __future__ import print_function
from __future__ import division

import logging
import numpy as np
from torch.utils.data.sampler import BatchSampler


class ClassBalancedSampler(BatchSampler):
    """
    Sampler that generates class balanced indices with classes chosen randomly.
    For example, choosing batch_size = 32 and nun_samples_per_class = 8
    will result in
    32 indices, which point to 8 samples from 32/8=4 randomly picked classes.
    """

    def __init__(self, dataset, batch_size=80, num_samples_per_class=4):
        self.duplicate_small_classes = True
        assert batch_size % num_samples_per_class == 0, \
                "batch size must be divisable by num_samples_per_class"
        self.targets = np.array(dataset.ys)
        self.C = list(set(self.targets))
        self.C_index = {
            c: np.where(self.targets == c)[0] for c in self.C}
        for c in self.C:
            np.random.shuffle(self.C_index[c])
        self.C_count = {c: 0 for c in self.C}
        self.count = 0
        self.num_classes = batch_size // num_samples_per_class
        self.num_samples_per_class = num_samples_per_class
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        is_not_enough_classes = len(self.C) < self.num_classes
        if is_not_enough_classes:
            logging.warn(('Not enough classes to sample batches: have={},'
                         'required={}').format(len(self.C), self.num_classes))
        while self.count + self.batch_size < len(self.dataset):
            C = np.random.choice(
                self.C, self.num_classes, replace=is_not_enough_classes
            )
            indices = []
            for class_ in C:
                if self.C_count[class_] + self.num_samples_per_class\
                   > len( self.C_index[class_]):
                    indices.extend(
                        np.random.choice(
                            self.C_index[class_],
                            self.num_samples_per_class,
                            replace = len(
                                self.C_index[class_] < \
                                        self.num_samples_per_class
                                )
                            )
                        )
                else:
                    indices.extend(
                        self.C_index[class_][self.C_count[class_]:
                        self.C_count[class_] + self.num_samples_per_class]
                    )
                self.C_count[class_] += self.num_samples_per_class
                if self.C_count[class_] + self.num_samples_per_class \
                        > len( self.C_index[class_]):
                    np.random.shuffle(self.C_index[class_])
                    self.C_count[class_] = 0
            yield indices
            self.count += self.num_classes * self.num_samples_per_class

    def __len__(self):
        return len(self.dataset) // self.batch_size

