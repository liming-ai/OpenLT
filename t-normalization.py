import os, sys, shutil
import torch

train_id = sys.argv[1]
epochs = sys.argv[2]
experiment_type = sys.argv[3]
norm_weights = sys.argv[4] # 0.3146 for iNaturalist, 0.6968 for ImageNet-LT
norm_weights = float(norm_weights)
test_gpu = sys.argv[5]
use_parallel = sys.argv[6] == "True" or sys.argv[6] == "1"
diversity_num = sys.argv[7]
diversity_num = int(diversity_num)

print("train_id:", train_id, "epochs:", epochs, "experiment_type:", experiment_type)

os.makedirs('saved/{}/τ-norm/'.format(experiment_type))
shutil.copyfile('saved/{}/config.json'.format(experiment_type), 'saved/{}/τ-norm/config.json'.format(experiment_type))
shutil.copyfile('saved/{}/model_best.pth'.format(experiment_type), 'saved/{}/τ-norm/raw_model.pth'.format(experiment_type))

pth = torch.load('saved/{}/model_best.pth'.format(experiment_type))

def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

print(pth['state_dict'].keys())

def parallel(key):
    if use_parallel:
        return "module." + key
    else:
        return key

for ind in range(diversity_num):
    # import pdb; pdb.set_trace()
    if diversity_num == 1:
        linear_name = "linear"
    else:
        linear_name = f"linears.{ind}"

    weights = pth['state_dict'][parallel(f'model.{linear_name}.weight')].cpu()
    bias = pth['state_dict'][parallel(f'model.{linear_name}.bias')].cpu()

    ws = pnorm(weights, norm_weights)
    bs = bias * 0

    pth['state_dict'][parallel(f'model.{linear_name}.weight')] = ws
    pth['state_dict'][parallel(f'model.{linear_name}.bias')] = bs

torch.save(pth, 'saved/{}/τ-norm/τ-norm.pth'.format(experiment_type))
os.system("python test.py -d {} -r saved/{}/τ-norm/τ-norm.pth".format(test_gpu, experiment_type))
