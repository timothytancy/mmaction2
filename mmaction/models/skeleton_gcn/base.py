# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
# import logging
# logging.basicConfig(filename='sample_output1.log', level=logging.DEBUG)

class SoftTargetHandler():
    def __init__(self, mu):
        self.mu = mu
        self.cur_epoch_out = None
        self.ema = None
        self.is_last_iter = False
        self.epoch = 1

    def register(self, val):
        self.cur_epoch_out = torch.unsqueeze(val.clone(), 0)

    def concat_output(self, x):
        x_ = x.clone()
        # if x not same size as self.output, pad tensor to match size. 
        # this is done so that we can stack outputs from the whole epoch by iteration
        if x_.size()[0] != self.cur_epoch_out.size()[1]:
            self.is_last_iter = True
            tgt_len = self.cur_epoch_out.size()[1]
            x_ = self.pad_to_len(x_, tgt_len)
        x_ = torch.unsqueeze(x_, 0)

        # add latest batch output to cur_epoch_out (stacking along dim 0)
        prev_out = self.cur_epoch_out.clone()
        self.cur_epoch_out = torch.cat((prev_out, x_), 0)

    def pad_to_len(self, tensor, tgt_len):
        num_classes = self.cur_epoch_out.size()[-1]
        padding = torch.zeros(tgt_len - tensor.size()[0], num_classes).cuda()
        return torch.cat((tensor, padding), dim=0)
        

    def get_iter_num(self):
        return self.cur_epoch_out.size()[0]
    
    def roll_window(self):
        if self.ema is None:
            self.ema = self.cur_epoch_out
        else:
            new_average = (1.0 - self.mu) * self.cur_epoch_out + self.mu * self.ema
            self.ema = new_average
        self.cur_epoch_out = None
        self.is_last_iter = False
        self.update_epoch(self.epoch + 1)
    
    def update_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, idx):
        return self.ema[idx]
        
class BaseGCN(nn.Module, metaclass=ABCMeta):
    """Base class for GCN-based action recognition.

    All GCN-based recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature.
            Default: None.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head=None, train_cfg=None, test_cfg=None):
        super().__init__()
        # record the source of the backbone
        self.backbone_from = 'mmaction2'
        self.backbone = builder.build_backbone(backbone)
        self.cls_head = builder.build_head(cls_head) if cls_head else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.use_soft_tgts = self.train_cfg["use_soft_tgts"]
        if self.use_soft_tgts:
            self.sth = SoftTargetHandler(0.5)
            self.burn_in = self.train_cfg["burn_in"]
            self.temperature = self.train_cfg["temperature"]
            self.beta = self.train_cfg["beta"]
            self.gamma = self.train_cfg["gamma"]

        self.init_weights()

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        if self.backbone_from in ['mmcls', 'mmaction2']:
            self.backbone.init_weights()
        else:
            raise NotImplementedError('Unsupported backbone source '
                                      f'{self.backbone_from}!')

        if self.with_cls_head:
            self.cls_head.init_weights()

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        """Defines the computation performed at training."""

    @abstractmethod
    def forward_test(self, *args):
        """Defines the computation performed at testing."""

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            train = self.forward_train(keypoint, label, **kwargs)
            # logging.debug(f"forward_train: {train}")  #{'top1_acc': tensor(0.0156, dtype=torch.float64), 'top5_acc': tensor(0.0781, dtype=torch.float64), 'loss_cls': tensor(6.1427, grad_fn=<MulBackward0>)}
            return train

        test = self.forward_test(keypoint, **kwargs)
        # logging.debug(f"forward_test: {test}")
        return test

    def extract_feat(self, skeletons):
        """Extract features through a backbone.

        Args:
            skeletons (torch.Tensor): The input skeletons.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(skeletons)
        # logging.debug(f"extracting features: f{x.size()}")
        return x

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        skeletons = data_batch['keypoint']
        label = data_batch['label']
        # logging.debug(f"training skeletons size: {skeletons.size()}")

        label = label.squeeze(-1)

        # added this
        label = F.one_hot(label, num_classes=self.cls_head.num_classes)
        losses = self(skeletons, label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(skeletons.data))
        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        skeletons = data_batch['keypoint']
        label = data_batch['label']
        # logging.debug(f"validation label: {label}")
        # added this
        label = F.one_hot(label, num_classes=self.cls_head.num_classes)

        losses = self(skeletons, label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(skeletons.data))

        return outputs
