
import tensorflow as tf
import math
from tensorflow.python.eager import context
from tensorflow.python.training import optimizer
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops 
from tensorflow.python.keras import backend_config 
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 
from tensorflow.python.ops import array_ops 
from tensorflow.python.ops import control_flow_ops 
from tensorflow.python.ops import math_ops 
from tensorflow.python.ops import state_ops 
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops 
from tensorflow.python.util.tf_export import keras_export
from tensorflow import clip_by_value
import numpy as np
import math

class Sampler(object):
	def __init__(self, n, p = None, d = None):
		"""
		@param n: number of samples
		@param p: probability of assigning Y=1, p > 1/2
		@param d: data dimension d >= 6n
		"""
		assert p is None or p > 0.5
		assert d is None or d > 6*n
		self.n = n
		if p is None:
			p = 0.0
			while p <= 0.5:
				p = np.random.rand()
		self.p = p
		self.d = 6*n+1 if d is None else d

	def sample(self):
		probs = np.random.rand(self.n)
		Y = np.ones(self.n).astype(np.int)
		neg_idx = np.where(probs > self.p)
		Y[neg_idx] = 0
		X = np.zeros((self.n, self.d), np.float)
		X[:,0] = Y
		X[:,[1,2]] = 1
		for i in range(self.n):
			start_idx = 3+5*i
			end_idx = start_idx + 2*(1-Y[i]) + 1
			X[i,start_idx:end_idx] = 1
		return X, Y
class Frankenstein(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate  = 0.001,
        epsilon=1e-8,
        weight_decay=0.0, 
        fixed_beta=0.0,
        name = "Frankenstein",
        **kwargs,):
        super(Frankenstein,self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("weight_decay", weight_decay)
        self.epsilon = epsilon or backend_config.epsilon()
        self._set_hyper("fixed_beta", fixed_beta)
        self._has_fix_beta = fixed_beta != 0.0
        self._has_weight_decay = weight_decay != 0.0
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m",initializer='zeros')
        for var in var_list:
            self.add_slot(var, "s",initializer=tf.keras.initializers.Constant(self.epsilon))
        for var in var_list:
            self.add_slot(var, "vmax",initializer='zeros')

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        m = self.get_slot(var, "m")
        s = self.get_slot(var, "s")
        vmax = self.get_slot(var, "vmax")
        epsilon_t=tf.convert_to_tensor(self.epsilon, var_dtype)
        lr_t =tf.convert_to_tensor(self.learning_rate, var_dtype)
        wd_t = tf.convert_to_tensor(self.weight_decay, var_dtype)
        fixed_beta = tf.convert_to_tensor(self.fixed_beta, var_dtype)
        
        if self._has_fix_beta:
            momentum_t=fixed_beta
        else:
            momentum_t = math_ops.subtract(1.0,
                                           tf.clip_by_value(math_ops.multiply(
                                               math_ops.sqrt(math_ops.divide(lr_t,1e-3)),0.1)
                                               ,0.05,0.5))         
        v_f=math_ops.divide(math_ops.acos(math_ops.tanh(math_ops.multiply(m,grad))),math.pi)
        pen=math_ops.add(math_ops.square(grad) , epsilon_t)
        vmax_t=math_ops.maximum(vmax, pen)
        kk=math_ops.exp(math_ops.negative(math_ops.abs(math_ops.subtract(v_f,s))))
        dfc =math_ops.divide( 1.60653065971,math_ops.add(1.0,kk))
        lr_t1=math_ops.multiply(math_ops.divide(lr_t,(math_ops.sqrt(vmax_t))),dfc)
        temp1=math_ops.subtract(0.5,v_f)
        temp2=math_ops.multiply(grad , lr_t1)
        new_m=math_ops.multiply(m,
                                math_ops.log(math_ops.add(2.71828182846,
                                             tf.clip_by_value(math_ops.add(temp1,math_ops.sqrt(pen))
                                                              ,-1.71828182846,0.1393692896))))
        
        new_m =math_ops.subtract(math_ops.multiply(momentum_t ,new_m) , temp2)
        new_p = math_ops.subtract(temp2, math_ops.multiply(momentum_t,new_m))
        gg=math_ops.multiply(math_ops.divide(pen,s),math_ops.abs(temp1))
        new_vmax=math_ops.add(math_ops.multiply(vmax_t,math_ops.subtract(1.0,gg)),math_ops.multiply(pen,gg))
        
        updata_m=m.assign(new_m, use_locking=self._use_locking)
        updata_s=s.assign(pen, use_locking=self._use_locking)
        updata_vamx=vmax.assign(new_vmax, use_locking=self._use_locking)
        
        if self._has_weight_decay:
            new_p+=wd * var*lr_t1
        
        var_update = var.assign_sub(new_p, use_locking=self._use_locking)
        updates = [var_update, updata_m, updata_s,updata_vamx]
        return tf.group(*updates)
    def _resource_apply_sparse(self, grad, var):
        var_dtype = var.dtype.base_dtype
        m = self.get_slot(var, "m")
        s = self.get_slot(var, "s")
        vmax = self.get_slot(var, "vmax")
        epsilon_t=tf.convert_to_tensor(self.epsilon, var_dtype)
        lr_t =tf.convert_to_tensor(self.learning_rate, var_dtype)
        wd_t = tf.convert_to_tensor(self.weight_decay, var_dtype)
        fixed_beta = tf.convert_to_tensor(self.fixed_beta, var_dtype)
        
        if self._has_fix_beta:
            momentum_t=fixed_beta
        else:
            momentum_t = math_ops.subtract(1.0,
                                           tf.clip_by_value(math_ops.multiply(
                                               math_ops.sqrt(math_ops.divide(lr_t,1e-3)),0.1)
                                               ,0.05,0.5))         
        v_f=math_ops.divide(math_ops.acos(math_ops.tanh(math_ops.multiply(m,grad))),math.pi)
        pen=math_ops.add(math_ops.square(grad) , epsilon_t)
        vmax_t=math_ops.maximum(vmax, pen)
        kk=math_ops.exp(math_ops.negative(math_ops.abs(math_ops.subtract(v_f,s))))
        dfc =math_ops.divide( 1.60653065971,math_ops.add(1.0,kk))
        lr_t1=math_ops.multiply(math_ops.divide(lr_t,(math_ops.sqrt(vmax_t))),dfc)
        temp1=math_ops.subtract(0.5,v_f)
        temp2=math_ops.multiply(grad , lr_t1)
        scale_m=math_ops.log(math_ops.add(2.71828182846,
                                             tf.clip_by_value(math_ops.add(temp1,math_ops.sqrt(pen))
                                                              ,-1.71828182846,0.1393692896)))
        new_m = state_ops.assign(m, math_ops.multiply(m , scale_m),use_locking=self._use_locking)
        with ops.control_dependencies([new_m]):
            new_m = self._resource_scatter_add(m, indices, m_scaled_g_values)
            
        new_p = math_ops.subtract(temp2, math_ops.multiply(momentum_t,new_m))
        gg=math_ops.multiply(math_ops.divide(pen,s),math_ops.abs(temp1))
        
        scale_vmax=math_ops.subtract(1.0,gg)
        scale_plus=math_ops.multiply(pen,gg)
        
        new_vmax = state_ops.assign(vmax, vmax_t * scale_vmax,
                           use_locking=self._use_locking)
        with ops.control_dependencies([new_vmax]):
            new_vmax = self._resource_scatter_add(vmax, indices, scale_plus)
        
        if self._has_weight_decay:
            new_p+=wd * var*lr_t1
            
        updata_s=s.assign(pen, use_locking=self._use_locking)
        var_update = var.assign_sub(new_p, use_locking=self._use_locking)
        updates = [var_update, new_m, updata_s,new_vmax]
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "epsilon": self._serialize_hyperparameter("epsilon"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "fixed_beta": self._serialize_hyperparameter("fixed_beta"),
            }
        )
        return config



class AdaBound(optimizer_v2.OptimizerV2):
    
    def __init__(self, 
                 learning_rate=1e-3, 
                 beta_1=0.9, 
                 beta_2=0.999, 
                 final_lr=0.1, 
                 gamma=1e-3, 
                 epsilon=1e-8, 
                 amsbound=False, 
                 name='AdaBound', 
                 **kwargs):
        super(AdaBound, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr',learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('final_lr', final_lr)
        self._set_hyper('gamma', gamma)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsbound = amsbound
        self.base_lr = learning_rate
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsbound:
            for var in var_list:
                self.add_slot(var, 'vhat')
                
    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaBound, self)._prepare_local(var_device, var_dtype, apply_state)
        
        local_step = math_ops.cast(self.iterations +1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        gamma_power = math_ops.pow(self._get_hyper('gamma', var_dtype), local_step)
        lr = apply_state[(var_device, var_dtype)]['lr_t'] * \
              ((math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        final_lr = math_ops.multiply(self._get_hyper('final_lr', var_dtype),
                    math_ops.divide(apply_state[(var_device, var_dtype)]['lr_t'], 
                                 ops.convert_to_tensor(self.base_lr, var_dtype)))
        apply_state[(var_device, var_dtype)].update(dict(
            lr=lr, 
            epsilon=ops.convert_to_tensor(self.epsilon, var_dtype), 
            gamma_power=gamma_power,
            final_lr=final_lr,
            beta_1_t=beta_1_t, 
            one_minus_beta_1_t=1 - beta_1_t, 
            beta_2_t=beta_2_t, 
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t
        ))
        
    def set_weights(self, weights):
        params = self.weights
        
        num_vars = int((len(params) -1) /2)
        if len(weights) == 3 * num_vars +1:
            weights = weights[:len(params)]
        super(AdaBound, self).set_weights(weights)
        
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype 
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        
        
        

        lower_bound = coefficients['final_lr'] * (1. - 1. / (coefficients['gamma_power'] + 1.))
        upper_bound = coefficients['final_lr'] * (1. + 1. / (coefficients['gamma_power']))
        
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m*coefficients['beta_1_t'] + m_scaled_g_values, use_locking=self._use_locking)
        
        
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'] + v_scaled_g_values, use_locking=self._use_locking)
        
        if self.amsbound:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat), use_locking=self._use_locking)
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            v_sqrt = math_ops.sqrt(v_t)
        
        step_size_bound = coefficients['lr'] / (v_sqrt + coefficients['epsilon'])
        bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)
        
        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)
        
        if self.amsbound:
            return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])
        return control_flow_ops.group(*[var_update, m_t, v_t])
    

    def _resource_apply_sparse(self, grad, var, indcs, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype 
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        
        
        

        lower_bound = coefficients['final_lr'] * (1. - 1. / (coefficients['gamma_power'] + 1.))
        upper_bound = coefficients['final_lr'] * (1. + 1. / (coefficients['gamma_power']))
        
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m*coefficients['beta_1_t'] , use_locking=self._use_locking, name='assign_m_t')
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indcs, m_scaled_g_values)
        
        
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'] , use_locking=self._use_locking, name='assign_v_t')
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indcs, v_scaled_g_values )
        
        if self.amsbound:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat), use_locking=self._use_locking)
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            v_sqrt = math_ops.sqrt(v_t)
        
        step_size_bound = coefficients['lr'] / (v_sqrt + coefficients['epsilon'])
        bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)
        
        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)
        
        
        if self.amsbound:
            return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])
        return control_flow_ops.group(*[var_update, m_t, v_t])
    
    
    
    def get_config(self):
        config = super(AdaBound, self).get_config()
        config.update({
            'learning_rate' : self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'gamma': self._serialize_hyperparameter('gamma'),
            'final_lr': self._serialize_hyperparameter('final_lr'), 
            'epsilon': self.epsilon, 
            'amsbound': self.amsbound,
        })
        return config



