
import torchvision
import torch
from math import ceil
import logging
import torch
import numpy as np
import torchvision
import torch
from torch.nn import Linear, Dropout, AvgPool2d, MaxPool2d
from torch.nn.init import xavier_normal_


def resnet50(pretrained = True):
    model = torchvision.models.resnet50(pretrained = pretrained)

    model.features = torch.nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4
    )

    model.sz_features_output = 2048

    for module in filter(
        lambda m: type(m) == torch.nn.BatchNorm2d, model.modules()
    ):
        module.eval()
        module.train = lambda _: None

    return model


def make_parameters_dict(model, filter_module_names):
    """
    Separates model parameters into 'backbone' and other modules whose names
    are given in as list in `filter_module_names`, e.g. ['embedding_layer'].
    """

    # init parameters dict
    D = {k: [] for k in ['backbone', *filter_module_names]}
    for name, param in model.named_parameters():
        name = name.split('.')[0]
        if name not in filter_module_names:
            D['backbone'] += [param]
        else:
            D[name] += [param]

    # verify that D contains same number of parameters as in model
    nb_total = len(list(model.parameters()))
    nb_dict_params = sum([len(D[d]) for d in D])
    assert nb_total == nb_dict_params
    return D


def init_splitted(layer, nb_clusters, sz_embedding):
    # initialize splitted embedding parts separately
    from math import ceil
    for c in range(nb_clusters):
        i = torch.arange(
            c * ceil(sz_embedding / nb_clusters),
            # cut off remaining indices, e.g. if > embedding size
            min(
                (c + 1) * ceil(
                    sz_embedding / nb_clusters
                ),
                sz_embedding
            )
        ).long()
        _layer = torch.nn.Linear(layer.weight.shape[1], len(i))
        layer.weight.data[i] = xavier_normal_(_layer.weight.data, gain = 1)
        layer.bias.data[i] = _layer.bias.data


def embed_model(model, config, sz_embedding, normalize_output=True):

    model.features_pooling = AvgPool2d(7,
        stride=1, padding=0, ceil_mode=True, count_include_pad=True
    )
    model.features_dropout = Dropout(0.01)

    # choose arbitrary parameter for selecting GPU/CPU
    dev = list(model.parameters())[0].device
    if type(model) != torchvision.models.ResNet:
        model.sz_features_output = _sz_features[type(model)]
    torch.random.manual_seed(config['random_seed'] + 1)
    model.embedding = Linear(model.sz_features_output, sz_embedding).to(dev)

    # for fair comparison between different cluster sizes
    torch.random.manual_seed(config['random_seed'] + 1)
    np.random.seed(config['random_seed'] + 1)

    init_splitted(
        model.embedding, config['nb_clusters'], config['sz_embedding']
    )

    features_parameters = model.features.parameters()

    model.parameters_dict = make_parameters_dict(
        model = model,
        filter_module_names = ['embedding']
    )

    assert normalize_output

    nb_clusters = config['nb_clusters']

    learner_neurons = [None] * nb_clusters
    for c in range(nb_clusters):
        learner_neurons[c] = np.arange(
            c * ceil(sz_embedding / nb_clusters),
            # cut off remaining indices, e.g. if > embedding size
            min(
                (c + 1) * ceil(
                    sz_embedding / nb_clusters
                ),
                sz_embedding
            )
        )
    model.learner_neurons = learner_neurons

    def forward(x, use_penultimate=False):
        x = model.features(x)
        x = model.features_pooling(x)
        x = model.features_dropout(x)
        x = x.view(x.size(0), -1)
        if not use_penultimate:
            x = model.embedding(x)
            for idxs in model.learner_neurons:
                x[:, idxs] = torch.nn.functional.normalize(
                    x[:, idxs], p=2, dim=1
                )
        else:
            # normalize the entire penultimate layer
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
    model.forward = forward


def make(config):
    model = resnet50(pretrained = True)
    embed_model(
        model = model,
        config = config,
        sz_embedding = config['sz_embedding'],
        normalize_output = True
    )
    return model
