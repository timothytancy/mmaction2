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
        
        # if first iteration of epoch
        if self.ema.cur_epoch_out is None:
            self.ema.register(output)
        else:
            #if not first iter of new epoch:
            self.ema.concat_output(output)

        # access outputs ema by slicing by iteration number
        # soft_outputs = self.ema(iter_num)  # HOW TO GET ITER NUM???
        logging.debug(f"current epoch output size: {self.ema.cur_epoch_out.size()}")
        
        # convert labels to one-hot
        gt_labels = F.one_hot(gt_labels, num_classes=self.cls_head.num_classes)

        # # soften labels
        # gt_labels = gt_labels * 0.6 + output * 0.4  
        loss = self.cls_head.loss(output, gt_labels)
        losses.update(loss)

        # if we are in the first iter of new epoch:
        if self.ema.is_last_iter:
            logging.debug(f"rolling window... ")
            self.ema.roll_window()

        return losses

    def forward_test(self, skeletons):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output.data.cpu().numpy()
    
