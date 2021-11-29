import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, Sampler
from model.model import *
from data_loader.cifar_data_loaders import *

from sklearn.manifold import TSNE
from matplotlib import cm



SEED = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
torch.set_printoptions(2, sci_mode=False)
np.set_printoptions(2, suppress=True)

num_classes = 10
epoch = 200  # 20 40 80 120 160 200
print(epoch)
dataset = 'CIFAR10'
method = 'CE_epoch_{}'.format(epoch)
IR = 100
model_path = 'saved/CIFAR10/LT_100/R32_SupCon_Cosine_classifier_StepLR_160_180_StandardAug_128_0.1_5e-4_200e_DRW_160/models/model_best.pth'

print(torch.load(model_path)['monitor_best'])
model = ResNet32Model(num_classes, use_norm=True, returns_feat=True).cuda()
model.load_state_dict(torch.load(model_path)['state_dict'])

train_loader = ImbalanceCIFAR10DataLoader('data', 1024, training=True, imb_factor=1.0/IR)
test_loader =  ImbalanceCIFAR10DataLoader('data', 1024, training=False, imb_factor=1.0/IR)


with torch.no_grad():
    model.eval()
    for batch_idx, data in enumerate(train_loader):
        data, target = data
        data, target = data.cuda(), target.cuda()
        output = model(data)
        output, feat = output['output'], output['feat']

        if batch_idx == 0:
            train_feat = feat
            train_label = target
        else:
            train_feat = torch.cat([train_feat, feat], dim=0)
            train_label = torch.cat([train_label, target], dim=0)

    for batch_idx, data in enumerate(test_loader):
        data, target = data
        data, target = data.cuda(), target.cuda()
        output= model(data)
        output, feat = output['output'], output['feat']

        if batch_idx == 0:
            test_feat = feat
            test_label = target
        else:
            test_feat = torch.cat([test_feat, feat], dim=0)
            test_label = torch.cat([test_label, target], dim=0)


train_emb = TSNE(n_iter=1000, perplexity=50, early_exaggeration=15, init='pca', random_state=0).fit_transform(train_feat.cpu())
test_emb = TSNE(n_iter=1000, perplexity=50, early_exaggeration=15, init='pca', random_state=0).fit_transform(test_feat.cpu())


color_cycle = ["darkorange", "deeppink", "blue", "brown", "red", "dimgrey", "gold", "green", "darkturquoise","blueviolet"]


plt.cla()

X, Y = train_emb[:, 0], train_emb[:, 1]
# In order to make the color distinguishable, the 0-255 color interval is divided into 9 points
# and then the label is mapped to an interval
for x, y, s in zip(X, Y, train_label):
    c = cm.rainbow(int(255 / 9 * s))
    plt.scatter(x, y, s=8, color=color_cycle[int(s)])

plt.xticks([])
plt.yticks([])
plt.title('Training Datset(IR {})'.format(IR))
plt.savefig("imgs/cifar10_train_{}.jpg".format(method), dpi=1080)

plt.cla()

X, Y = test_emb[:, 0], test_emb[:, 1]
# In order to make the color distinguishable, the 0-255 color interval is divided into 9 points
# and then the label is mapped to an interval
for x, y, s in zip(X, Y, test_label):
    c = cm.rainbow(int(255 / 9 * s))
    plt.scatter(x, y, s=8, color=color_cycle[int(s)])

plt.xticks([])
plt.yticks([])
plt.title('Testing Datset(IR {})'.format(IR))
plt.savefig("imgs/cifar10_test_{}.jpg".format(method), dpi=1080)