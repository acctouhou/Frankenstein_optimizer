import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
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
                 weight_decay=0, weight_decouple=True, fixed_beta=0,base_lr=1e-3,base_beta=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= fixed_beta < 1.0:
            raise ValueError("Invalid momentum value: {}".format(fixed_beta))
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay,
                        weight_decouple=weight_decouple,
                        fixed_beta=fixed_beta,base_lr=base_lr,base_beta=base_beta)

        super(Frankenstein, self).__init__(params, defaults)
        
        self.max_xi=float(np.exp(1.03))
        self.min_xi=float(np.exp(-0.2))
        self.max_beta_adj=float(0.05)
        self.pi=float(math.pi)
        
        
        
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
                    beta_1=group['fixed_beta']
                else:
                    
                    beta_1=1.0- max(self.max_beta_adj, min(1-self.max_beta_adj, (1-group['base_beta']) * math.sqrt(group['lr'] / group['base_lr'])))
                
                if group['weight_decay'] > 0:
                    if group['weight_decouple']:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p.data, alpha=group['weight_decay'])
                
                
                p_factor=torch.div(torch.acos(torch.tanh(torch.mul(m,grad))),self.pi)  # frankenstein
                xi =torch.div(1.60653065971,torch.add(1.0,torch.exp(-torch.abs(torch.add(s ,-p_factor)))))

                x_t=torch.add(torch.mul(grad,grad) ,group['eps'])
                v_t=torch.max(vmax, x_t)
                sqrt_v=torch.sqrt(v_t)
                alpha_xi_sqrt_v=torch.mul(torch.div(group['lr'],sqrt_v),xi)
                rho_factor=torch.log(torch.clamp(3.21828182846-p_factor+sqrt_v, min=self.min_rho,max=self.max_rho))
                m.mul_(torch.mul(rho_factor,beta_1)).add_(torch.mul(-grad , alpha_xi_sqrt_v))                 # Momentum update
                beta_2=torch.mul(torch.clamp(torch.div(x_t,s),0.0,1.0),torch.abs(p_factor-0.5))  # actually, 1-beta_2
                p.data.add_(torch.add(torch.mul(beta_1,m),torch.mul(-grad, alpha_xi_sqrt_v))) # Parameter update
                vmax.copy_(torch.add(torch.mul(v_t,torch.add(1.0,-beta_2)),torch.mul(beta_2,x_t)))  # v_t update
                s.copy_(x_t)
        return loss
