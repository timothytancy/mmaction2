# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import RECOGNIZERS
from .base import BaseGCN
import torch
import torch.nn.functional as F
import numpy as np 
# import logging
# logging.basicConfig(filename='sample_output1.log', level=logging.DEBUG)


@RECOGNIZERS.register_module()
class SkeletonGCN(BaseGCN):
    """Spatial temporal graph convolutional networks."""
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        losses = dict()
        
        x = self.extract_feat(skeletons)
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)
        gt_labels = F.one_hot(gt_labels, num_classes=self.cls_head.num_classes)  # convert to one-hot labels

        if self.use_soft_tgts:
            # update soft target logs (for next epoch)
            mixed_output = gt_labels * 0.6 + self.soften_targets(output) * 0.4  # hardcoding weight temporarily

            # if first iteration of epoch
            if self.sth.cur_epoch_out is None:
                self.sth.register(mixed_output)
            else:
                self.sth.concat_output(mixed_output)           

            # block to execute after burn-in period
            if self.sth.epoch >= self.burn_in:
                assert self.sth.ema is not None, "Allow model to burn-in for at least one epoch (burn_in>=2) before softening targets"
                
                # for current epoch, target is made using predictions from previous epochs
                iter_num = self.sth.get_iter_num()  # iter_num is 1-indexed
                prev_out = self.sth.ema[iter_num-1]
                
                # for last iteration in epoch, chop previous padded saved outputs down to size
                if gt_labels.size() != prev_out.size():
                    prev_out = prev_out[:gt_labels.size()[0]]
                    
                gt_labels = gt_labels * 0.6 + prev_out * 0.4  # mix previous preds with label
            
            # if we are in the first iter of new epoch:
            if self.sth.is_last_iter:
                # logging.debug(f"rolling window... ")
                self.sth.roll_window()

        # compute loss of hard outputs against soft labels
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
    
    def soften_targets(self, x):
        """
        Takes in a tensor as input, and outputs softened distribution as an np array.
        """
        x_np = x.detach().numpy()
        x_exp = np.exp(x_np/self.temperature)
        x_sum = np.repeat(np.sum(x_exp, axis=1)[:, np.newaxis], self.cls_head.num_classes, axis=1) # sum over class dim then repeat again to match num_classes
        out = x_exp/x_sum
        
        return torch.from_numpy(out)
        # return torch.from_numpy(out).requires_grad_()
    
