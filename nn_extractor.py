# A feature extractor for NN. A reimplementation of the architecture in "Property Inference Attacks on FCNN"


import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
from collections import OrderedDict


class Phi(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Phi, self).__init__()
        self.linear = Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = x.transpose(1,2)
        x = self.bn(x)
        return torch.tanh(x).transpose(1,2)
        

class Extractor(nn.Module):
    def __init__(self, sample_params, rep_dims, feature_dims):
        super(Extractor, self).__init__()
        # construct the networks (both are of the layer size)
        assert len(rep_dims) == (len(sample_params) // 2)
        self.phi_modules = []
        self.rep_dims = rep_dims
        self.VERBOSE = False
        self.dims = []
        # self.feature_dim = feature_dim
        for i in range(len(sample_params)//2):
            input_dim = sample_params[2*i].shape[1] + 1 # (feature concat bias)
            self.dims.append(sample_params[2*i].shape[1])
            if(i > 0):
                input_dim += self.rep_dims[i-1] # * sample_params[2*i].shape[1]
            self.phi_modules.append(Phi(input_dim, self.rep_dims[i]))
            print(sample_params[2*i].shape)
            print(sample_params[2*i+1].shape)
        self.phi_modules = nn.ModuleList(self.phi_modules)
        self.dims.append(sample_params[-1].shape[0])
        print(self.phi_modules)
        
        # construct the rho module (an MLP)
        self.rho = []
        input_dim = np.sum(self.rep_dims)
        for i, output_dim in enumerate(feature_dims):
            self.rho.append(("fc_{}".format(i), Linear(input_dim, output_dim)))
            # self.rho.append(("bn_{}".format(i), nn.BatchNorm1d(output_dim)))
            # if(i < len(feature_dims) - 1):
            self.rho.append(("sigmoid_{}".format(i), nn.Tanh()))
            input_dim = output_dim
        self.rho = nn.Sequential(OrderedDict(self.rho))
        print(self.rho)
        print(self.dims)

    def split(self, params):
        # @param: params are of the shape [batch_size, total_parameter_num]
        """
torch.Size([128, 3072])
torch.Size([128])
torch.Size([32, 128])
torch.Size([32])
torch.Size([16, 32])
torch.Size([16])
torch.Size([10, 16])
torch.Size([10])

[3072, 128, 32, 16, 10]
        """
        param_list = []
        start = 0
        for i in range(len(self.dims)-1):
            weight_size = self.dims[i] * self.dims[i+1]
            param_list.append(params[:, start:start+weight_size].reshape(-1, self.dims[i+1], self.dims[i]))
            start += weight_size
            bias_size =  self.dims[i+1]
            param_list.append(params[:, start:start+bias_size].reshape(-1, self.dims[i+1]))
        return param_list

    def forward(self, params):
        params = self.split(params)
        layer_num = len(params) // 2
        layer_features = []
        for i in range(layer_num):
            weight = params[2*i]
            bias = params[2*i+1]
            layer_feature = torch.cat((weight, bias.unsqueeze(2)), dim = 2)
            if(i > 0):
                # context = torch.cat([layer_features[-1][i, :] for i in range(layer_features[-1].size(0))], dim = 0)
                context =  torch.mean(layer_features[-1], dim = 1)
                context = context.unsqueeze(1).repeat_interleave(layer_feature.size(1), dim = 1)
                layer_feature = torch.cat((layer_feature, context), dim = 2)
            # do batch normalization over it
            layer_feature = self.phi_modules[i](layer_feature)
            # print("Layer {}:{}".format(i, layer_feature.shape))
            layer_features.append(layer_feature)
            
        for i in range(layer_num):
            layer_features[i] = torch.mean(layer_features[i], dim = 1)
            # print(layer_features[i].shape)
            if(self.VERBOSE):
                print("Layer {}'s  Feature: {}".format(i+1, layer_features[i]))
        nn_feature = torch.cat(layer_features, dim = 1)
        if(self.VERBOSE):
            print("Concatenated  Feature: {}".format(nn_feature))
        nn_feature = self.rho(nn_feature)
        # print("Final Feature: {}".format(nn_feature.shape))
        if(self.VERBOSE): print("Final Feature: {}".format(nn_feature))
        
        return nn_feature
    
        
        
        

