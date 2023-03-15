# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import RECOGNIZERS
from .base import BaseGCN
import torch
import torch.nn.functional as F
import numpy as np 
import logging
logging.basicConfig(filename='sample_output1.log', level=logging.DEBUG)


@RECOGNIZERS.register_module()
class SkeletonGCN(BaseGCN):
    """Spatial temporal graph convolutional networks."""
    
    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        losses = dict()
        
        x = self.extract_feat(skeletons)
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)

        if self.use_soft_tgts:
            # update soft target logs (for next epoch)
            mixed_output = gt_labels * self.beta + self.soften_targets(output, self.temperature) * (1-self.beta) 

            # if first iteration of epoch
            if self.sth.cur_epoch_out is None:
                self.sth.register(mixed_output)
            else:
                self.sth.concat_output(mixed_output)           

            # block to execute after burn-in period
            if self.sth.epoch >= self.burn_in:
                ERR_MSG = "Allow model to burn-in for at least one epoch (burn_in>=2) before softening targets"

                assert self.sth.ema is not None, ERR_MSG
                
                # for current epoch, target is made using predictions from previous epochs
                iter_num = self.sth.get_iter_num()  # iter_num is 1-indexed
                prev_out = self.sth.ema[iter_num-1]
                
                # for last iteration in epoch, chop previous padded saved outputs down to size
                if gt_labels.size() != prev_out.size():
                    prev_out = prev_out[:gt_labels.size()[0]]

                gt_labels = gt_labels * self.gamma + prev_out * (1-self.gamma)  # mix previous preds with label
            
            # if we are in the first iter of new epoch:
            if self.sth.is_last_iter:
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
    
    def soften_targets(self, x, temperature):
        """
        Takes in a tensor as input, and outputs softened distribution as an np array.
        """
        x_np = x.detach().cpu().numpy()
        x_exp = np.exp(x_np/temperature)
        x_sum = np.repeat(np.sum(x_exp, axis=1)[:, np.newaxis], self.cls_head.num_classes, axis=1) # sum over class dim then repeat again to match num_classes
        out = x_exp/x_sum
        return torch.from_numpy(out).cuda()

        # x_exp = torch.exp(x/self.temperature)
        # x_sum = torch.sum(x_exp, dim=1).unsqueeze(1)
        # x_sum_classes = x_sum.repeat(1, self.cls_head.num_classes)
        # out = x_exp/x_sum_classes
        # return out

        # return torch.from_numpy(out).requires_grad_()
    
