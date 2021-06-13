from __future__ import absolute_import
from __future__ import print_function

import torch
from torch.nn.utils.clip_grad import clip_grad_norm



def init_lr_scheduler(optimizer,
                      lr_scheduler='multi_step',  # learning rate scheduler
                      stepsize=[20, 40],  # step size to decay learning rate
                      gamma=0.1  # learning rate decay
                      ):
    if lr_scheduler == 'single_step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize[0], gamma=gamma)

    elif lr_scheduler == 'multi_step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    else:
        raise ValueError('Unsupported lr_scheduler: {}'.format(lr_scheduler))

def GCN_lr_scheduler(optimizer, model, grad_clip):
    params = list(model.parameters())
    if grad_clip > 0:
        clip_grad_norm(params, grad_clip)
