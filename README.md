
# OpenLT: An open-source project for long-tail classification
## Supported Methods for Long-tailed Recognition:
- [x] Cross-Entropy Loss
- [x] [Focal Loss (ICCV'17)](https://arxiv.org/abs/1708.02002)
- [x] [Class-Balanced Re-weightin (CVPR'19)](https://arxiv.org/abs/1901.05555)
- [x] [LDAM Loss (NIPS'19)](https://arxiv.org/abs/1906.07413)
- [x] [Balance Softmax Loss (NIPS'20)](https://arxiv.org/abs/2007.10740)
- [x] [Decouple (ICLR'20)](https://arxiv.org/abs/1910.09217): cRT
- [x] [Decouple (ICLR'20)](https://arxiv.org/abs/1910.09217): tau-normalization
- [x] [Decouple (ICLR'20)](https://arxiv.org/abs/1910.09217): LWS
- [x] [RIDE (ICLR'21)](https://arxiv.org/abs/2010.01809)
- [x] [Logit Adjustment Loss (ICLR'21)](https://arxiv.org/abs/2007.07314)
- [x] [DisAlign (CVPR'21)](https://arxiv.org/abs/2103.16370)
- [x] [Bayias Loss (NIPS'21)](https://arxiv.org/abs/2111.03874)
- [x] [Adaptive Logit Adjustment Loss (arXiv'21)](https://arxiv.org/abs/2104.06094)


## Reproduce Results
Here we simply show part of results to prove that our implementation is reasonable.
### ImageNet-LT
| Method | Backbone | Reported Result | Our Implementation |
| :----:| :----: | :----: | :----: |
| CE | ResNet-10 | 34.8 | **35.3** |
| Decouple-cRT | ResNet-10 | 41.8 | **41.8** |
| Decouple-LWS | ResNet-10 | 41.4 | **41.6** |
| BalanceSoftmax | ResNet-10 | **41.8** | 41.4 |
| CE | ResNet-50 | 41.6 | **43.2** |
| LDAM-DRW* | ResNet-50 | 48.8 | **51.2** |
| Decouple-cRT | ResNet-50 | 47.3 | **48.7** |
| Decouple-LWS | ResNet-50 | 47.7 | **49.3** |

### CIFAR100-LT (Imbalance Ratio 100)
${\dagger}$ means the reported results are copied from [LADE](https://arxiv.org/abs/2012.00321)
| Method | Datatset | Reported Result | Our Implementation |
| :----:| :----: | :----: | :----: |
| CE | CIFAR100-LT | 39.1 | **40.3** |
| LDAM-DRW | CIFAR100-LT | 42.04 | **42.9** |
| LogitAdjust | CIFAR100-LT | 43.89 | **45.3** |
| BalanceSoftmax$^{\dagger}$ | CIFAR100-LT | 45.1 | **46.47** |
| MiSLAS | CIFAR100-LT | 47 | **47.38** |
## Requirement
### Packages
* Python >= 3.7, < 3.9
* PyTorch >= 1.6
* tqdm (Used in `test.py`)
* tensorboard >= 1.14 (for visualization)
* pandas
* numpy
## Dataset Preparation
CIFAR code will download data automatically with the dataloader. We use data the same way as [classifier-balancing](https://github.com/facebookresearch/classifier-balancing). For ImageNet-LT and iNaturalist, please prepare data in the `data` directory. ImageNet-LT can be found at [this link](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf). iNaturalist data should be the 2018 version from [this](https://github.com/visipedia/inat_comp) repo (Note that it requires you to pay to download now). The annotation can be found at [here](https://github.com/facebookresearch/classifier-balancing/tree/master/data). Please put them in the same location as below:
```
data
├── cifar-100-python
│   ├── file.txt~
│   ├── meta
│   ├── test
│   └── train
├── cifar-100-python.tar.gz
├── ImageNet_LT
│   ├── ImageNet_LT_open.txt
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_val.txt
│   ├── Tiny_ImageNet_LT_train.txt (Optional)
│   ├── Tiny_ImageNet_LT_val.txt (Optional)
│   ├── Tiny_ImageNet_LT_test.txt (Optional)
│   ├── test
│   ├── train
│   └── val
└── iNaturalist18
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_val.txt
    └── train_val2018
```

## Training and Evaluation Instructions
#### Single Stage Training
```
python train.py -c path_to_config_file
```
For example, to train a model with LDAM Loss on CIFAR-100-LT:
```
python train.py -c configs/CIFAR-100/LDAMLoss.json
```
#### Decouple Training (Stage-2)
```
python train.py -c path_to_config_file -crt path_to_stage_one_checkpoints
```
For example, to train a model with LWS classifier on ImageNet-LT:
```
python train.py -c configs/ImageNet-LT/R50_LWS.json -lws path_to_stage_one_checkpoints
```
<!--
##### cRT (load from a checkpoint without linear and freezes the pretrained parameters)
This part is not finalized and will probably change.
```
python train.py --load_crt path_to_cRT_checkpoint -c path_to_config --reduce_dimension 1 --num_experts 3
```
##### t-norm
This part is not finalized and will probably change.
Please see `t-normalization.py` for usages. It requires a hyperparemeter from the decouple paper.
-->
### Test
To test a checkpoint, please put it with the corresponding config file.
```
python test.py -r path_to_checkpoint
```
### resume
```
python train.py -c path_to_config_file -r path_to_resume_checkpoint
```
Please see [the pytorch template that we use](https://github.com/victoresque/pytorch-template) for additional more general usages of this project

## FP16 Training
If you set fp16 in [utils/util.py](utils/util.py), it will enable fp16 training. However, this is susceptible to change (and may not work on all settings or models) and please double check if you are using it since we don't plan to focus on this part if you request help. Only some models work (see autograd in the code). We do not plan to provide support on this because it is not within our focus (just for faster training and less memory requirement).
In our experiments, the use of FP16 training does not reduce the accuracy of the model, regardless of whether it is a small dataset (CIFAR-LT) or a large dataset(ImageNet_LT, iNaturalist).


## Visualization
We use tensorboard as a visualization tool, and provide the accuracy changes of each class and different groups during the training process:
```
tensorboard --logdir path_to_dir
```

We also provide the simple code to visualize feature distribution using t-SNE and calibration using the reliability diagrams, please check the parameters in [plot_tsne.py](plot_tsne.py) and [plot_ece.py](plot_ece.py), and then run:
```
python plot_tsne.py
```
or
```
python plot_ece.py
```

## Pytorch template
This is a project based on this [pytorch template](https://github.com/victoresque/pytorch-template). The readme of the template explains its functionality, although we try to list most frequently used ones in this readme.

### License
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
This project is mainly based on [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition)'s code base. In the process of reproducing and organizing the code, it also refers to some other excellent code repositories, such as [decouple](https://github.com/facebookresearch/classifier-balancing) and [LDAM](https://github.com/kaidic/LDAM-DRW).