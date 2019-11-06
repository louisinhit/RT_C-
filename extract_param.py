import numpy
import numpy as np
import sys, os
import torch
import pickle
device = torch.device("cuda")
model = torch.load('../playwith2018.01dataset/saved_model/' + \
    'new.r_d_64_h_2_t_LSTM_ksize_8_level_2_n_2_lr_0.0001_dropout_0.01.pt')
model.to(device)

tp = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        tp[(name)] = param.data.cpu().numpy()
        print (name, tp[(name)].shape)
# just have a check
print (tp['rt.feed_forward.w_1.weight'].shape)
# write to .dat file
'''
wb = open('param.dat', 'wb')
pickle.dump(tp, wb)
wb.close()

# read from file for check
pkl = open('param.dat', 'rb')
mydict = pickle.load(pkl)
pkl.close()
# double check
#for i in mydict['linear.weight']:
#    print (i)
print (mydict['linear.weight'].shape)
print (mydict['rt.forward_net.1.feed_forward.w_2.bias'].shape)
    '''