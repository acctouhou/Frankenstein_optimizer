

# NN library
import torch
import torch.nn as nn
import torch.optim as optim
# Bert model and its tokenizer
from transformers import BertTokenizer, BertModel
# Text data
from torchtext import data, datasets
# Numerical computation
import numpy as np
# standard library
import random
import time
# Configuration
# Training configurations
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] =sys.argv[1]
tar_opt=int(sys.argv[2])
import math
tlr=2e-4
SEED = 9487
TRAIN = True
BATCH_SIZE = 128 
N_EPOCHS = 50


# Architecture
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
# Set random seed for reproducible experiments
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from torch.optim.optimizer import Optimizer, required
from collections import defaultdict
class AdaBound(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

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
# Get tokens for BERT

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
init_token_id = tokenizer.cls_token_id
eos_token_id  = tokenizer.sep_token_id
pad_token_id  = tokenizer.pad_token_id
unk_token_id  = tokenizer.unk_token_id

max_input_len = tokenizer.max_model_input_sizes['bert-base-uncased']

# Tokensize and crop sentence to 510 (for 1st and last token) instead of 512 (i.e. `max_input_len`)
def tokenize_and_crop(sentence):
  tokens = tokenizer.tokenize(sentence)
  tokens = tokens[:max_input_len - 2]
  return tokens

# Load the IMDB dataset and
# return (train_iter, valid_iter, test_iter) tuple
def load_data():
  text = data.Field(
    batch_first=True,
    use_vocab=False,
    tokenize=tokenize_and_crop,
    preprocessing=tokenizer.convert_tokens_to_ids,
    init_token=init_token_id,
    pad_token=pad_token_id,
    unk_token=unk_token_id
  )

  label = data.LabelField(dtype=torch.float)

  train_data, test_data  = datasets.IMDB.splits(text, label)
  train_data, valid_data = train_data.split(random_state=random.seed(SEED))

  print(f"training examples count: {len(train_data)}")
  print(f"test examples count: {len(test_data)}")
  print(f"validation examples count: {len(valid_data)}")

  label.build_vocab(train_data)

  train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
  )

  return train_iter, valid_iter, test_iter

# Get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 
# Build model
# 
from transformers import BertModel, BertConfig

bert_model = BertModel.from_pretrained('bert-base-uncased')
#configuration = BertConfig()
#bert_model=BertModel(configuration)



# Sentiment model containing pretrained BERT as backbone
# and two-GRU layers for analyzing the BERT hidden representation
# and a linear layer for classfification (the sigmoid is applied by the criterion during training).
import torch.nn as nn

class SentimentModel(nn.Module):
  def __init__(
    self,
    bert,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout
  ):
      
    super(SentimentModel, self).__init__()
    
    self.bert = bert
    embedding_dim = bert.config.to_dict()['hidden_size']
    self.rnn = nn.GRU(
      embedding_dim,
      hidden_dim,
      num_layers=n_layers,
      bidirectional=bidirectional,
      batch_first=True,
      dropout=0 if n_layers < 2 else dropout
    )
    self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, text):
    with torch.no_grad():
      embedded = self.bert(text)[0]
            
    _, hidden = self.rnn(embedded)
    
    if self.rnn.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
    else:
      hidden = self.dropout(hidden[-1,:,:])
    
    output = self.out(hidden)
    return output

model = SentimentModel(
  bert_model,
  HIDDEN_DIM,
  OUTPUT_DIM,
  N_LAYERS,
  BIDIRECTIONAL,
  DROPOUT
)
from pytorch_ranger import Ranger
ls_opt=[Frankenstein(model.parameters(), lr=tlr),
            torch.optim.Adam(model.parameters(), lr=tlr,amsgrad=False),
            torch.optim.Adam(model.parameters(), lr=tlr,amsgrad=True),
            torch.optim.SGD(model.parameters(), lr=tlr,momentum=0.9,nesterov=True),
            torch.optim.RMSprop(model.parameters(), lr=tlr),
            Lookahead(torch.optim.Adam(model.parameters(), lr=tlr,amsgrad=False),),
            AdaBound(model.parameters(), lr=tlr,final_lr=0.1),                                                            
            Ranger(model.parameters(), lr=tlr)]
#print(model)

# time taken for single epoch
def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

# computes accuracy
def binary_accuracy(preds, y):
  rounded_preds = torch.round(torch.sigmoid(preds))
  correct = (rounded_preds == y).float()
  acc = correct.sum() / len(correct)
  return acc

# training step
def train(model, iterator, optimizer, criterion):
  # stats
  epoch_loss = 0
  epoch_acc = 0
  # train mode
  model.train()
  
  for batch in iterator:
    # train step
    optimizer.zero_grad()
    predictions = model(batch.text).squeeze(1)
    loss = criterion(predictions, batch.label)
    acc = binary_accuracy(predictions, batch.label)
    loss.backward()
    optimizer.step()
    # stats
    epoch_loss += loss.item()
    epoch_acc += acc.item()
  
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

# evaluates the model on given iterator (either 
# train_iter, valid_iter, or test_iter)
def evaluate(model, iterator, criterion):
    
  epoch_loss = 0
  epoch_acc = 0
  # evaluation mode
  model.eval()
  
  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.text).squeeze(1)
      loss = criterion(predictions, batch.label)
      acc = binary_accuracy(predictions, batch.label)
      epoch_loss += loss.item()
      epoch_acc += acc.item()
      
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

# function to make sentiment prediction during inference
def predict_sentiment(model, tokenizer, sentence):
  model.eval()
  tokens = tokenizer.tokenize(sentence)
  tokens = tokens[:max_input_len - 2]
  indexed = [init_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_id]
  tensor = torch.LongTensor(indexed).to(device)
  tensor = tensor.unsqueeze(0)
  prediction = torch.sigmoid(model(tensor))
  return prediction.item()
log=[]
if __name__ == "__main__":
  # Train BERT
  if TRAIN:
    # load data
    train_iter, valid_iter, test_iter = load_data()

    #optimizer = optim.Adam(model.parameters())
    optimizer = ls_opt[tar_opt]
    criterion = nn.BCEWithLogitsLoss().to(device)
    model = model.to(device)
    
    best_val_loss = float('inf')

    for epoch in range(N_EPOCHS):
      # start time
      start_time = time.time()
      # train for an epoch
      train_loss, train_acc = train(model, train_iter, optimizer, criterion)
      valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
      test_loss, test_acc = evaluate(model, test_iter, criterion)
      log.append([train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc])
      # end time
      np.savetxt('pretrain_%d.txt'%(tar_opt),np.array(log))
      end_time = time.time()
      # stats
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      # save model if has validation loss
      # better than last one
      # stats
