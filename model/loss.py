import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

eps = 1e-7

# CE and LDAM are supported

# If you would like to add other losses, please have a look at:
# Focal Loss: https://github.com/kaidic/LDAM-DRW
# CRD, PKT, and SP Related Part: https://github.com/HobbitLong/RepDistiller

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, gamma=0., reweight_epoch=-1):
        """
        Paper: Focal loss for dense object detection (ICCV 2017)
        arXiv: https://arxiv.org/abs/1708.02002
        Source Code: https://github.com/kaidic/LDAM-DRW
        """
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.reweight_epoch = reweight_epoch

        if reweight_epoch != -1:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights_enabled = None
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def forward(self, output_logits, target):
        return focal_loss(F.relu(F.cross_entropy(output_logits, target, reduction='none', weight=self.per_cls_weights)), self.gamma)


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights = None
        else:
            self.reweight_epoch = reweight_epoch
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def forward(self, output_logits, target): # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)


class LDAMLoss(nn.Module):
    """
    Paper: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (NIPS 2019)
    arXiv: https://arxiv.org/abs/1906.07413
    Source Code: https://github.com/kaidic/LDAM-DRW
    """
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class RIDELoss(nn.Module):
    """
    Paper: RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts (ICLR 2021)
    arXiv: https://arxiv.org/abs/2010.01809
    Source Code: https://github.com/frank-xwang/RIDE-LongTailRecognition
    """
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1,
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0

            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)

            base_diversity_temperature = self.base_diversity_temperature
            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature

            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)

            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')

        return loss


class RIDELossWithDistill(nn.Module):
    """
    Paper: RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts (ICLR 2021)
    arXiv: https://arxiv.org/abs/2010.01809
    Source Code: https://github.com/frank-xwang/RIDE-LongTailRecognition
    """
    def __init__(self, cls_num_list=None, additional_distill_loss_factor=1.0, distill_temperature=1.0, ride_loss_factor=1.0, **kwargs):
        super().__init__()
        self.ride_loss = RIDELoss(cls_num_list=cls_num_list, **kwargs)
        self.distill_temperature = distill_temperature

        self.ride_loss_factor = ride_loss_factor
        self.additional_distill_loss_factor = additional_distill_loss_factor

    def to(self, device):
        super().to(device)
        self.ride_loss = self.ride_loss.to(device)
        return self

    def _hook_before_epoch(self, epoch):
        self.ride_loss._hook_before_epoch(epoch)

    def forward(self, student, target=None, teacher=None, extra_info=None):
        output_logits = student
        if extra_info is None:
            return self.ride_loss(output_logits, target)

        loss = 0
        num_experts = len(extra_info['logits'])
        for logits_item in extra_info['logits']:
            loss += self.ride_loss_factor * self.ride_loss(output_logits, target, extra_info)
            distill_temperature = self.distill_temperature

            student_dist = F.log_softmax(student / distill_temperature, dim=1)
            with torch.no_grad():
                teacher_dist = F.softmax(teacher / distill_temperature, dim=1)

            distill_loss = F.kl_div(student_dist, teacher_dist, reduction='batchmean')
            distill_loss = distill_temperature * distill_temperature * distill_loss
            loss += self.additional_distill_loss_factor * distill_loss
        return loss


class ALALoss(nn.Module):
    """
    Paper: Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition (Preprint)
    arXiv: https://arxiv.org/abs/2104.06094
    """
    def __init__(self, cls_num_list=None, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = torch.from_numpy(np.array(cls_num_list))
            m_list = np.log(2) / torch.log(m_list / m_list.min() + 1)
            # m_list = 1.0 / torch.log(m_list / m_list.min() + 1)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):

        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        hardness_term = output_logits / self.s
        hardness_term = (1 - hardness_term) / 2.0 * index
        hardness_term = hardness_term.detach()

        adjust_term = self.m_list[target].unsqueeze(-1)
        adjust_term = adjust_term * index
        adjust_term = adjust_term.detach()

        final_output = x - hardness_term * adjust_term * self.s

        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class LogitAdjustLoss(nn.Module):
    """
    Paper: Long-Tail Learning via Logit Adjustment (ICLR 2021)
    arXiv: https://arxiv.org/abs/2007.07314
    Source Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    """
    def __init__(self, cls_num_list=None, reweight_epoch=-1, tau=1):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = torch.from_numpy(np.array(cls_num_list))
            self.m_list = m_list / m_list.sum()
            self.tau = tau

            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        adjust_term = (self.m_list.unsqueeze(0) + 1e-12).log() * self.tau
        adjust_term = adjust_term.detach()

        final_output = x + adjust_term

        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class BalancedSoftmaxLoss(nn.Module):
    """
    Paper: Balanced Meta-Softmax for Long-Tailed Visual Recognition (NIPS 2020)
    arXiv: https://arxiv.org/abs/2007.10740
    Source Code: https://github.com/jiawei-ren/BalancedMetaSoftmax
    """
    def __init__(self, cls_num_list=None, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            self.m_list = torch.from_numpy(np.array(cls_num_list))

            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        adjust_term = (self.m_list.unsqueeze(0) + 1e-12).log()
        adjust_term = adjust_term.detach()

        final_output = x + adjust_term

        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class LabelAwareSmoothingLoss(nn.Module):
    def __init__(self, cls_num_list=None, smooth_head=0.4, smooth_tail=0.1, shape='concave', power=None):
        """
        Paper: Improving Calibration for Long-Tailed Recognition (CVPR 2021)
        arXiv: https://arxiv.org/abs/2104.00466
        Source Code: https://github.com/dvlab-research/MiSLAS/blob/0e6a47e01c0878c13ba5e1587fdf0887c1bf2f0b/methods.py#L31
        """
        super().__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth).float()

    def to(self, device):
        super().to(device)

        self.smooth = self.smooth.to(device)

        return self

    def forward(self, output_logits, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(output_logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()


class DisAlignLoss(nn.Module):
    """
    Paper: Distribution Alignment: A Uniﬁed Framework for Long-tail Visual Recognition (CVPR 2021)
    arXiv: https://arxiv.org/abs/2103.16370
    Source Code: https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/modeling/losses/grw_loss.py
    """
    def __init__(self, cls_num_list=None, p=1.5):
        super().__init__()

        if cls_num_list is None:
            self.m_list = None
            self.per_cls_weights = None
        else:
            self.m_list = torch.from_numpy(np.array(cls_num_list))

            self.per_cls_weights = self.m_list / self.m_list.sum()  # r_c in paper
            self.per_cls_weights = 1.0 / self.per_cls_weights
            self.per_cls_weights = self.per_cls_weights ** p
            self.per_cls_weigths = self.per_cls_weights / self.per_cls_weights.sum() * len(cls_num_list)

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target):
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)


class BayiasLoss(nn.Module):
    """
    Paper: Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective (NIPS 2021)
    arXiv: https://arxiv.org/abs/2111.03874
    OpenReview: https://openreview.net/forum?id=vqzAfN-BoA_
    """
    def __init__(self, cls_num_list=None):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            m_list = torch.from_numpy(np.array(cls_num_list))
            self.m_list = m_list / m_list.sum()

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        return self

    def get_final_output(self, output_logits, target):
        x = output_logits

        adjust_term = (self.m_list.unsqueeze(0) + 1e-12).log() + np.log(len(self.m_list))
        adjust_term = adjust_term.detach()

        final_output = x + adjust_term

        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


# TODO
# class InfluenceBalancedLoss(nn.Module):
#     def __init__(self, cls_num_list=None, reweight_epoch=-1, alpha=100):
#         super().__init__()
#         if cls_num_list is None:
#             self.m_list = None
#         else:
#             self.reweight_epoch = reweight_epoch
#             if reweight_epoch != -1:
#                 idx = 1 # condition could be put in order to set idx
#                 betas = [0, 0.9999]
#                 effective_num = 1.0 - np.power(betas[idx], cls_num_list)
#                 per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
#                 per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
#                 self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
#             else:
#                 self.per_cls_weights = None

#     def to(self, device):
#         super().to(device)

#         if self.per_cls_weights is not None:
#             self.per_cls_weights = self.per_cls_weights.to(device)

#         return self

#     def _hook_before_epoch(self, epoch):

#         if self.reweight_epoch != -1:
#             self.epoch = epoch

#             if epoch > self.reweight_epoch:
#                 self.per_cls_weights = self.per_cls_weights
#             else:
#                 self.per_cls_weights = None

#     def forward(self, output_logits, target):
#         if self.m_list is None:
#             return F.cross_entropy(output_logits, target)


#         return F.cross_entropy(final_output, target, weight=self.per_cls_weights)