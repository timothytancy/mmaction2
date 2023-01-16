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
        logging.debug(f"current epoch: {self.sth.epoch}, burn-in: {self.burn_in}")
        
        x = self.extract_feat(skeletons)
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)  # these are not one-hot labels 
        
        # block to execute after burn-in period
        if self.sth.epoch >= self.burn_in:
            assert self.sth.ema is not None, "Allow model to burn-in for at least one epoch (burn_in>=2) before softening targets"
            # if first iteration of epoch
            if self.sth.cur_epoch_out is None:
                self.sth.register(output)
            else:
                self.sth.concat_output(output)

            logging.debug(f"logging epochs: {self.sth.cur_epoch_out.size()}")

            # extract results from previous epochs
            iter_num = self.sth.get_iter_num()  # iter_num is 1-indexed
            gt_labels = F.one_hot(gt_labels, num_classes=self.cls_head.num_classes)  # one-hot labels
            prev_out = self.sth.ema[iter_num]
            gt_labels = gt_labels * 0.6 + prev_out * 0.4  

        loss = self.cls_head.loss(output, gt_labels)
        losses.update(loss)

        # if we are in the first iter of new epoch:
        if self.sth.is_last_iter:
            logging.debug(f"rolling window... ")
            self.sth.roll_window()

        return losses

    def forward_test(self, skeletons):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output.data.cpu().numpy()
    
