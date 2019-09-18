# A feature extractor for NN. A reimplementation of the architecture in "Property Inference Attacks on FCNN"


import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
from collections import OrderedDict

class Extractor(nn.Module):
    def __init__(self, sample_params, rep_dims, feature_dims):
        super(Extractor, self).__init__()
        # construct the networks (both are of the layer size)
        assert len(rep_dims) == (len(sample_params) // 2)
        self.phi_modules = []
        self.rep_dims = rep_dims
        # self.feature_dim = feature_dim
        for i in range(len(sample_params)//2):
            input_dim = sample_params[2*i].shape[1] + 1 # (feature concat bias) 
            if(i > 0):
                input_dim += self.rep_dims[i-1] * sample_params[2*i].shape[1]
            self.phi_modules.append(nn.Sequential(
                Linear(input_dim, self.rep_dims[i]),
                nn.Sigmoid()
            ))
        self.phi_modules = nn.ModuleList(self.phi_modules)
        print(self.phi_modules)
        # construct the rho module (an MLP)
        self.rho = []
        input_dim = np.sum(self.rep_dims)
        for i, output_dim in enumerate(feature_dims):
            self.rho.append(("fc_{}".format(i), Linear(input_dim, output_dim)))
            if(i < len(feature_dims) - 1):
                self.rho.append(("sigmoid_{}".format(i), nn.Sigmoid()))
            input_dim = output_dim
        self.rho = nn.Sequential(OrderedDict(self.rho))
        print(self.rho)
        

    def forward(self, params):
        layer_num = len(params) // 2
        layer_features = []
        for i in range(layer_num):
            weight = params[2*i]
            bias = params[2*i+1]
            layer_feature = torch.cat((weight, bias.unsqueeze(1)), dim = 1)
            if(i > 0):
                context = torch.cat([layer_features[-1][i, :] for i in range(layer_features[-1].size(0))], dim = 0)
                context = context.unsqueeze(0).repeat_interleave(layer_feature.size(0), dim = 0)
                layer_feature = torch.cat((layer_feature, context), dim = 1)
            layer_feature = self.phi_modules[i](layer_feature)
            layer_features.append(layer_feature)
        for i in range(layer_num):
            layer_features[i] = torch.sum(layer_features[i], dim = 0)
        nn_feature = torch.cat(layer_features)
        nn_feature = self.rho(nn_feature)
        return nn_feature
    
        
        
            
            
            
            
        # @input: the structured parameters of an nn
        # @output: a vector-form feature
        pass
        
        
        

