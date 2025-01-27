import math
import torch
from torch.optim.optimizer import Optimizer

class Frankenstein(Optimizer):
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
                 weight_decay=0, weight_decouple=True, fixed_beta=0):
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
                lr_t=torch.mul(torch.div(group['lr'],torch.sqrt(temp1)),dfc)
                temp2=torch.log(torch.clamp(2.71828182846-v_f+0.5
                , 0.81873075307,2.8010658347))
                m.mul_(torch.mul(temp2,momentum)).add_(torch.mul(-grad , lr_t))
                temp3=torch.mul(torch.clamp(torch.div(pen,s),0.0,1.0),torch.abs(v_f-0.5))
                p.data.add_(torch.add(torch.mul(momentum,m),torch.mul(-grad, lr_t)))
                vmax.copy_(torch.add(torch.mul(temp1,torch.add(1,-temp3)),torch.mul(temp3,pen)))
                s.copy_(pen)
        return loss