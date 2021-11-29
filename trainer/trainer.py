import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16, mixup_data, calibration
import model.model as module_arch

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, num_classes, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, lr_scheduler, config)
        self.config = config
        self.num_classes = num_classes
        self.distill = config._config.get('distill', False)
        self.mixup = config._config.get('mixup', False)
        self.cls_num_list = data_loader.cls_num_list
        self.using_balanced_sampler = data_loader.balanced

        # class balance sampler
        if self.using_balanced_sampler:
            self.epoch_steps = int(sum(data_loader.cls_num_list) / data_loader.batch_size)

        # mixup
        if self.mixup:
            self.logger.info("using mixup {}".format(self.mixup))
            self.mixup_alpha = self.mixup['alpha'] if 'alpha' in self.mixup else 0
            self.mixup_start_epoch = self.mixup['start_epoch'] if 'start_epoch' in self.mixup else 0
            self.mixup_end_epoch = self.mixup['end_epoch'] if 'end_epoch' in self.mixup else float('inf')

        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)

        # distill
        if self.distill:
            self.logger.info("** Distill is on, please double check distill_checkpoint in config **")
            self.teacher_model = config.init_obj('distill_arch', module_arch)
            teacher_checkpoint = torch.load(config['distill_checkpoint'], map_location="cpu")

            self.teacher_model = self.teacher_model.to(self.device)

            teacher_state_dict = teacher_checkpoint["state_dict"]

            rename_parallel_state_dict(teacher_state_dict)

            if len(self.device_ids) > 1:
                self.logger.info("Using multiple GPUs for teacher model")
                self.teacher_model = torch.nn.DataParallel(self.teacher_model, device_ids=self.device_ids)
                load_state_dict(self.teacher_model, {"module." + k: v for k, v in teacher_state_dict.items()}, no_ignore=True)
            else:
                load_state_dict(self.teacher_model, teacher_state_dict, no_ignore=True)

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=None)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=None)

        train_cls_num_list = np.array(data_loader.cls_num_list)
        self.idx_many_shot = train_cls_num_list > 100
        self.idx_medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        self.idx_few_shot = train_cls_num_list < 20

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.real_model._hook_before_iter()
        self.train_metrics.reset()
        self.writer.set_step(epoch - 1, 'train')

        confusion_matrix = torch.zeros(self.num_classes, self.num_classes).cuda()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        for batch_idx, data in enumerate(self.data_loader):
            # Break when step equal to epoch step
            # Please refer to https://github.com/facebookresearch/classifier-balancing/blob/f162adef63fdf592d0b8a0f6e52bc481fc08785f/run_networks.py#L265
            if self.using_balanced_sampler and batch_idx == self.epoch_steps:
                break

            if self.distill and len(data) == 4:
                data, target, idx, contrast_idx = data
                idx, contrast_idx = idx.to(self.device), contrast_idx.to(self.device)
            else:
                data, target = data
                idx, contrast_idx = None, None

            data, target = data.to(self.device), target.to(self.device)

            if self.mixup and self.mixup_start_epoch < epoch < self.mixup_end_epoch:
                data, target_a, target_b, lam = mixup_data(data, target, alpha=self.mixup_alpha)

            self.optimizer.zero_grad()

            with autocast():
                if self.real_model.requires_target:
                    output = self.model(data, target=target)
                    output, loss = output
                else:
                    extra_info = {}
                    output = self.model(data)
                    if self.distill:
                        with torch.no_grad():
                            teacher = self.teacher_model(data)
                            if idx is not None: # Contrast
                                extra_info.update({
                                    "idx": idx,
                                    "contrast_idx": contrast_idx
                                })
                            if isinstance(output, dict): # New return that does support DataParallel
                                feat_students = output["feat"]
                                extra_info.update({
                                    "feat_students": feat_students,
                                })
                                if isinstance(teacher, dict):
                                    feat_teachers = teacher["feat"]
                                    extra_info.update({
                                        "feat_teachers": feat_teachers,
                                    })
                            else: # Old return that does not support DataParallel
                                extra_info.update({
                                    "feat_students": self.real_model.model.feat,
                                    "feat_teachers": self.teacher_model.model.feat,
                                    "feat_students_before_GAP": self.real_model.model.feat_before_GAP,
                                    "feat_teachers_before_GAP": self.teacher_model.model.feat_before_GAP,
                                })
                        if isinstance(teacher, dict):
                            teacher = teacher["output"]
                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                            extra_info.update({
                                "logits": logits.transpose(0, 1)
                            })
                        else:
                            extra_info.update({
                                "logits": self.real_model.model.logits
                            })

                    if isinstance(output, dict):
                        feat = output["feat"]
                        output = output["output"]

                    if self.distill:
                        if self.mixup and self.mixup_start_epoch < epoch < self.mixup_end_epoch:
                            loss = lam * self.criterion(student=output, target=target_a, teacher=teacher, extra_info=extra_info) + \
                                (1 - lam) * self.criterion(student=output, target=target_b, teacher=teacher, extra_info=extra_info)
                        else:
                            loss = self.criterion(student=output, target=target, teacher=teacher, extra_info=extra_info)
                    elif self.add_extra_info:
                        if self.mixup and self.mixup_start_epoch < epoch < self.mixup_end_epoch:
                            loss = lam * self.criterion(output_logits=output, target=target_a, extra_info=extra_info) + \
                                (1 - lam) * self.criterion(student=output, target=target_b, extra_info=extra_info)
                        else:
                            loss = self.criterion(output_logits=output, target=target, extra_info=extra_info)
                    else:
                        if self.mixup and self.mixup_start_epoch < epoch < self.mixup_end_epoch:
                            loss = lam * self.criterion(output_logits=output, target=target_a) + \
                                (1 - lam) * self.criterion(output_logits=output, target=target_b)
                        else:
                            loss = self.criterion(output_logits=output, target=target)

            # FP16 Training
            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, return_length=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.4f} max group LR: {:.4f} min group LR: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self._update_confusion_matrix(confusion_matrix, 'Train')

        if self.do_validation and epoch > self.val_start_epoch:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

        self.model.eval()
        self.valid_metrics.reset()
        self.writer.set_step(epoch - 1, 'val')

        # for calibration computation
        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            if hasattr(self.model, "confidence_model") and self.model.confidence_model:
                cumulative_sample_num_experts = torch.zeros((self.model.model.num_experts, ), device=self.device)
                num_samples = 0
                confidence_model = True
            else:
                confidence_model = False
            for _, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if confidence_model:
                    output, sample_num_experts = self.model(data)
                    num, count = torch.unique(sample_num_experts, return_counts=True)
                    cumulative_sample_num_experts[num - 1] += count
                    num_samples += data.size(0)
                else:
                    output = self.model(data)
                if isinstance(output, dict):
                    # feat = output["feat"]
                    output = output["output"]

                prob = torch.softmax(output, dim=1)
                confidence_part, pred_class_part = torch.max(prob, dim=1)
                confidence = np.append(confidence, confidence_part.detach().cpu().numpy())
                pred_class = np.append(pred_class, pred_class_part.detach().cpu().numpy())
                true_class = np.append(true_class, target.detach().cpu().numpy())

                loss = self.criterion(output_logits=output, target=target)

                for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                self.valid_metrics.update('loss', loss.mean().item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if confidence_model:
                print("Samples with num_experts:", *[('%.2f'%item) for item in (cumulative_sample_num_experts * 100 / num_samples).tolist()])

        self._update_confusion_matrix(confusion_matrix, 'Val')
        self._update_calibration(true_class, pred_class, confidence, num_bins=15, mode='Val')

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


    def _update_confusion_matrix(self, confusion_matrix, mode):
        metrics = self.train_metrics if mode == 'train' else self.valid_metrics

        acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
        acc = acc_per_class.detach().cpu().numpy()
        many_shot_acc = acc[self.idx_many_shot].mean()
        medium_shot_acc = acc[self.idx_medium_shot].mean()
        few_shot_acc = acc[self.idx_few_shot].mean()

        self.writer.add_scalar('{}_Accuracy/{}'.format(mode, 'all'), acc_per_class.mean().item())
        self.writer.add_scalar('{}_Accuracy/{}'.format(mode, 'many'), many_shot_acc)
        self.writer.add_scalar('{}_Accuracy/{}'.format(mode, 'medium'), medium_shot_acc)
        self.writer.add_scalar('{}_Accuracy/{}'.format(mode, 'few'), few_shot_acc)

        # for cls_idx, acc in enumerate(acc_per_class):
        #     self.writer.add_scalar('Class Acc({})/{}'.format(mode, cls_idx), acc.item())

        self.writer.add_scalar('Loss/{}'.format(mode), metrics.avg('loss'))


    def _update_calibration(self, true_class, pred_class, confidence, num_bins=15, mode='val'):
        cal = calibration(true_class, pred_class, confidence, num_bins=15)
        self.writer.add_scalar('{}_Calibration/avg_accuracy'.format(mode), cal["avg_accuracy"])
        self.writer.add_scalar('{}_Calibration/avg_confidence'.format(mode), cal["avg_confidence"])
        self.writer.add_scalar('{}_Calibration/ece'.format(mode), cal["expected_calibration_error"])
        self.writer.add_scalar('{}_Calibration/mce'.format(mode), cal["max_calibration_error"])

        # idx_many = np.where(self.idx_many_shot[true_class.astype(np.int32)] == True)[0]
        # idx_mid = np.where(self.idx_medium_shot[true_class.astype(np.int32)] == True)[0]
        # idx_few = np.where(self.idx_few_shot[true_class.astype(np.int32)] == True)[0]

        # many_cal = calibration(true_class[idx_many], pred_class[idx_many], confidence[idx_many], num_bins=15)
        # mid_cal = calibration(true_class[idx_mid], pred_class[idx_mid], confidence[idx_mid], num_bins=15)
        # few_cal = calibration(true_class[idx_few], pred_class[idx_few], confidence[idx_few], num_bins=15)

        # for group, group_cal in zip(['many', 'medium', 'few'], [many_cal, mid_cal, few_cal]):
        #     self.writer.add_scalar('{}_Calibration/{}/avg_accuracy'.format(mode, group), group_cal["avg_accuracy"])
        #     self.writer.add_scalar('{}_Calibration/{}/avg_confidence'.format(mode, group), group_cal["avg_confidence"])
        #     self.writer.add_scalar('{}_Calibration/{}/ece'.format(mode, group), group_cal["expected_calibration_error"])
        #     self.writer.add_scalar('{}_Calibration/{}/mce'.format(mode, group), group_cal["max_calibration_error"])

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)