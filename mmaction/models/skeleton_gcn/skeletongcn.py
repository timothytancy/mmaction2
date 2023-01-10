# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import RECOGNIZERS
from .base import BaseGCN
import torch.nn.functional as F
import logging
logging.basicConfig(filename='sample_output1.log', level=logging.DEBUG)


@RECOGNIZERS.register_module()
class SkeletonGCN(BaseGCN):
    """Spatial temporal graph convolutional networks."""
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        losses = dict()
        logging.debug(f"current epoch: {self.current_epoch}")
        
        x = self.extract_feat(skeletons)
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)  # these are not one-hot labels 
        
        if "output" not in self.ema.shadow.keys():
            self.ema.register("output", output)
        

        ema_output = self.ema("output", output)
        # logger.debug(f"ema_output: {ema_output}")
        
        # convert labels to one-hot
        gt_labels = F.one_hot(gt_labels, num_classes=self.cls_head.num_classes)

        # soften labels
        gt_labels = gt_labels * 0.6 + output * 0.4  
        loss = self.cls_head.loss(output, gt_labels)
        losses.update(loss)

        return losses

    def forward_test(self, skeletons):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output.data.cpu().numpy()