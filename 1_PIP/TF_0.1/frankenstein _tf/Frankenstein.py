import tensorflow as tf
import math
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend_config

from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops


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
            self.add_slot(var, "s",initializer=tf.keras.initializers.Constant(self.learning_rate))
        for var in var_list:
            self.add_slot(var, "vmax",initializer='zeros')

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)
    def _resource_apply_dense(self, grad, var, indices, scatter_add):
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
            new_p+=wd_t * var*lr_t1
        
        var_update = var.assign_sub(new_p, use_locking=self._use_locking)
        updates = [var_update, updata_m, updata_s,updata_vamx]
        return tf.group(*updates)
    def _resource_apply_sparse(self, grad, var, indices, scatter_add):
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
            new_m = self._resource_scatter_add(m, indices, temp2)
            
        new_p = math_ops.subtract(temp2, math_ops.multiply(momentum_t,new_m))
        gg=math_ops.multiply(math_ops.divide(pen,s),math_ops.abs(temp1))
        
        scale_vmax=math_ops.subtract(1.0,gg)
        scale_plus=math_ops.multiply(pen,gg)
        
        new_vmax = state_ops.assign(vmax, vmax_t * scale_vmax,
                           use_locking=self._use_locking)
        with ops.control_dependencies([new_vmax]):
            new_vmax = self._resource_scatter_add(vmax, indices, scale_plus)
        
        if self._has_weight_decay:
            new_p+=wd_t * var*lr_t1
            
        updata_s=s.assign(pen, use_locking=self._use_locking)
        var_update = var.assign_sub(new_p, use_locking=self._use_locking)
        updates = [var_update, new_m, updata_s,new_vmax]
        return tf.group(*updates)
    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
            [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()
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
