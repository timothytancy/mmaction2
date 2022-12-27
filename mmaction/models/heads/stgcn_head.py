# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead
import logging
logging.basicConfig(filename='sample_output1.log', level=logging.DEBUG)


@HEADS.register_module()
class STGCNHead(BaseHead):
    """The classification head for STGCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        num_person (int): Number of person. Default: 2.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 num_person=2,
                 init_std=0.01,
                 soften_targets=False,
                 temperature = 1,
                 ema_window = 10,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_person = num_person
        self.init_std = init_std
        self.soften_targets = soften_targets
        self.temperature = temperature
        self.ema_window = ema_window

        self.pool = None
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError

        self.fc = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)
        self.tensor_queue = EMAPredictions(10, 0.9, torch.zeros([self.ema_window, 64, self.num_classes]))  
        # FIXME: hardcoded number of instances, but need to fix this later on

    def init_weights(self):
        normal_init(self.fc, std=self.init_std)

    def forward(self, x):
        # global pooling
        assert self.pool is not None
        x = self.pool(x)
        x = x.view(x.shape[0] // self.num_person, self.num_person, -1, 1,
                   1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.shape[0], -1)

        if self.soften_targets:
            x = self.soften_preds(x, ema, self.temperature)
        self.tensor_queue.enqueue_tensor(x)
        ema = self.tensor_queue.numpy_ewma_vectorized_v2()
            
        return x
    
    def soften_preds(self, x, ema, temperature):
        return torch.exp(x/temperature)/torch.exp(ema/temperature)

class EMAPredictions():
    def __init__(self, window, weight, queue):
        self.window = window
        self.weight = weight
        self.queue = queue
        assert self.window == self.queue.size(dim=0)

    def enqueue_tensor(self, x):
        # RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.
        self.queue[:-1] = self.queue[1:]
        self.queue[-1] = x
        return self.queue

    def numpy_ewma_vectorized_v2(self):
        data = self.queue.detach().numpy()
        alpha = 2 /(self.window + 1.0)
        alpha_rev = 1-alpha
        n = data.shape[0]

        pows = alpha_rev**(np.arange(n+1))

        scale_arr = 1/pows[:-1]
        offset = data[0]*pows[1:]
        pw0 = alpha*alpha_rev**(n-1)

        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        logging.debug(f"EMA queue size: {out.size()}")
        return out
            
            
    
        
    
        
        
