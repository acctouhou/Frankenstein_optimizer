
import numpy as np
import copy
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import scale
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow_addons as tfa

from utils import Sampler
from utils import Frankenstein
from utils import AdaBound


# setting 
batch_sizes=64
lr=1e-3
epochs=80
num_samples = 2000

# generate data 
s = Sampler(num_samples, 0.8)
X, y = s.sample()
XTrain, yTrain = X[:int(num_samples*0.7)], y[:int(num_samples*0.7)]
XTest, yTest = X[int(num_samples*0.7):], y[int(num_samples*0.7):]

# generate model 
def get_model(opt):
    inp=tf.keras.Input([X.shape[1]])
    x1=tf.keras.layers.Dense(1,activation='sigmoid',
                             kernel_initializer='zeros',
                             use_bias=False
                             )(inp)
    model=tf.keras.Model(inp,x1)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=opt,metrics=['accuracy'])
    return model



#training
log_acc=[]
log_loss=[]

for method in range(8):
    tf.keras.backend.clear_session()
    methods=[Frankenstein(learning_rate=lr),
                 tf.keras.optimizers.Adam(learning_rate=lr),
                 tf.keras.optimizers.Adam(learning_rate=lr,amsgrad=True),
                 tf.keras.optimizers.SGD(learning_rate=lr*10,momentum=0.9,nesterov=True),
                 tf.keras.optimizers.RMSprop(learning_rate=lr),
                  tfa.optimizers.Lookahead(tf.keras.optimizers.Adam(learning_rate=lr)),
                 AdaBound(learning_rate=lr),
                 tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(learning_rate=lr), sync_period=6, slow_step_size=0.5)
                ]
    
    model=get_model(methods[method])
    result=model.fit(XTrain, yTrain,
              batch_size=batch_sizes,epochs=epochs,validation_data=(XTest, yTest))
    log_acc.append(result.history['val_accuracy'])
    log_loss.append(result.history['val_loss'])





#plot result
#%%
cl=['#0C5DA5','aqua','#FF9500',
    '#845B97','#00B945','#FF2C00','olive']


fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(7,3),dpi=300)

for k in range(7):
    axes[0].plot(log_acc[k+1],c=cl[k],ls='--')
    axes[1].plot(log_loss[k+1],c=cl[k],ls='--')

axes[0].plot(log_acc[0],c='k',ls='--')
axes[1].plot(log_loss[0],c='k',ls='--')
axes[1].set_title('Testing Loss')
axes[0].set_title('Testing Accuracy')
axes[1].set_xlabel('Epoch')
axes[0].set_xlabel('Epoch')
plt.tight_layout()



