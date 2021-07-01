""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP
import numpy as np
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False
import math
import torch
from torch.optim.optimizer import Optimizer

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class Frankenstein (Optimizer):
    r"""Implements Frankenstein optimizer
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        fixed_beta (float, optional):  when fixed_beta!=0, the beta 
            is performed as a constant value
            when when fixed_beta==0, the beta depend on learning rate
            automatically (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
    """
    def __init__(self, params, lr=1e-3, eps=1e-8,
                 weight_decay=0, weight_decouple=False, fixed_beta=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= fixed_beta < 1.0:
            raise ValueError("Invalid momentum value: {}".format(fixed_beta))
        defaults = dict(lr=lr, eps=eps,
                        weight_decay=weight_decay, weight_decouple=weight_decouple,
                        fixed_beta=fixed_beta)
        super(Frankenstein, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(Frankenstein, self).__setstate__(state)
        
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Frankenstein does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format),group['lr'])
                    state['vmax'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m, s,vmax = state['m'], state['s'],state['vmax']
                
                
                if group['fixed_beta']!=0:
                    momentum=group['fixed_beta']
                else:
                    momentum=1.0-np.clip(0.1*math.sqrt(group['lr']/1e-3),0.05,0.5)
                
                if group['weight_decay'] > 0:
                    if group['weight_decouple']:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p.data, alpha=group['weight_decay'])
                v_f=torch.div(torch.acos(torch.tanh(torch.mul(m,grad))),3.14159)
                kk= torch.exp(-torch.abs(torch.add(s ,-v_f)))
                dfc =torch.div(1.60653065971,torch.add(1.0,kk))
                pen=torch.add(torch.mul(grad,grad) ,group['eps'])
                temp1=torch.max(vmax, pen)
                temp2=torch.sqrt(temp1)
                lr_t=torch.mul(torch.div(group['lr'],temp2)),dfc)
                temp3=torch.log(torch.clamp(3.21828182846-v_f+temp2
                , 0.81873075307,2.8010658347))
                m.mul_(torch.mul(temp3,momentum)).add_(torch.mul(-grad , lr_t))
                temp4=torch.mul(torch.clamp(torch.div(pen,s),0.0,1.0),torch.abs(v_f-0.5))
                p.data.add_(torch.add(torch.mul(momentum,m),torch.mul(-grad, lr_t)))
                vmax.copy_(torch.add(torch.mul(temp1,torch.add(1,-temp4)),torch.mul(temp4,pen)))
                s.copy_(pen)
        return loss
def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'Frankenstein':        
        optimizer = Frankenstein(parameters, **opt_args)  
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
