import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NormedLinear(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class LearnableWeightScalingLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, use_norm=False):
        super().__init__()
        self.classifier = NormedLinear(feat_dim, num_classes) if use_norm else nn.Linear(feat_dim, num_classes)
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.classifier(x) * self.learned_norm


class DisAlignLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, use_norm=False):
        super().__init__()
        self.classifier = NormedLinear(feat_dim, num_classes) if use_norm else nn.Linear(feat_dim, num_classes)
        self.learned_magnitude = nn.Parameter(torch.ones(1, num_classes))
        self.learned_margin = nn.Parameter(torch.zeros(1, num_classes))
        self.confidence_layer = nn.Linear(feat_dim, 1)
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, x):
        output = self.classifier(x)
        confidence = self.confidence_layer(x).sigmoid()
        return (1 + confidence * self.learned_magnitude) * output + confidence * self.learned_margin


def get_classifier(classifier, feat_dim, num_classes, use_norm=False, **kwargs):
    if classifier == 'linear':
        return nn.Linear(feat_dim, num_classes)
    elif classifier == 'cosine':
        return NormedLinear(feat_dim, num_classes)
    elif classifier == 'LWS':
        return LearnableWeightScalingLinear(feat_dim, num_classes, use_norm)
    elif classifier == 'DisAlign':
        return DisAlignLinear(feat_dim, num_classes, use_norm)
    else:
        raise Exception('Not supported classifier: {}'.format(classifier))