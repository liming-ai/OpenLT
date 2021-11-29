import os
import torch
import random
import argparse
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np

from tqdm import tqdm
from model.model import *
from data_loader.cifar_data_loaders import *
from data_loader.imagenet_lt_data_loaders import *
from parse_config import ConfigParser
from utils import calibration, reliability_diagrams



deterministic = True
if deterministic:
    # fix random seeds for reproducibility
    SEED = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


model_paths = [
    'saved/ImageNet_LT/R10/LWS_R10_CE_Linear_classifier_CosLR_StandardAug_512_0.2_5e-4_90e/models/model_best.pth',
    'saved/ImageNet_LT/R10/LWS_R10_LDAM_Cosine_classifier_CosLR_StandardAug_512_0.2_5e-4_90e/models/model_best.pth',
    'saved/ImageNet_LT/R10/LWS_R10_LDAM_Cosine_classifier_CosLR_Mixup_alpha_0.2_512_0.2_5e-4_90e/models/model_best.pth',
    'saved/ImageNet_LT/R10/LWS_R10_AdaptiveLDAM_Cosine_classifier_CosLR_StandardAug_512_0.2_5e-4_90e/models/model_best.pth',
    'saved/ImageNet_LT/R10/LWS_R10_AdaptiveLDAM_Cosine_classifier_CosLR_Mixup_alpha_0.2_512_0.2_5e-4_90e/models/model_best.pth',
]

model_names = ['LWS', 'LDAM+LWS', 'LDAM+LWS+mixup', 'Ours', 'Ours+mixup']

num_classes = 1000
# linear_model = ResNet32Model(num_classes, reduce_dimension=False, use_norm=False, num_experts=1, returns_feat=True).cuda()
# cosine_model = ResNet32Model(num_classes, reduce_dimension=False, use_norm=True, num_experts=1, returns_feat=True).cuda()
linear_model = ResNet10Model(num_classes, LWS=True, use_norm=False, returns_feat=True).cuda()
cosine_model = ResNet10Model(num_classes, LWS=True, use_norm=True, returns_feat=True).cuda()

models = [cosine_model if 'Cosine' in path else linear_model for path in model_paths]

# data_loader = ImbalanceCIFAR100DataLoader('data', 512, shuffle=False, training=False)
data_loader = ImageNetLTDataLoader('data/ImageNet_LT', 512, shuffle=False, training=False, num_workers=16)


results = {}
for name, model, model_path in zip(model_names, models, model_paths):
    state_dict = torch.load(model_path)['state_dict']
    model = torch.nn.DataParallel(model) if num_classes > 100 else model
    model.load_state_dict(state_dict)
    model.eval()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            if isinstance(output, dict):
                feat = output["feat"]
                output = output["output"]

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.detach().cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.detach().cpu().numpy())
            true_class = np.append(true_class, target.detach().cpu().numpy())

    results[name] = {
        "true_labels": true_class,
        "pred_labels": pred_class,
        "confidences": confidence
    }


reliability_diagrams(results, num_bins=15, num_cols=5, save_fig=True)