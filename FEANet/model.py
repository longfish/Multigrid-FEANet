import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""Note that input of KNet and FNet model is a batch of images with dimension of (N, Cin, H, W), N is batch size, Cin is number of channels."""

class KNet(nn.Module):
    def __init__(self, mesh):
        super(KNet, self).__init__()
        self.nnode_edge = mesh.nnode_edge
        self.kernel_dict = mesh.kernel_dict
        self.n_channel = len(mesh.kernel_dict)
        self.convert_global_pattern(mesh.global_pattern_center)
        self.net1 = nn.Conv2d(in_channels=1,out_channels=self.n_channel,kernel_size=3, padding = 1, bias = False)
        self.net2 = nn.Conv2d(in_channels=self.n_channel,out_channels=1,kernel_size=3, padding = 1, bias = False)
        for pkey in self.kernel_dict:
            K_weights = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]) # Identity tensor
            self.net1.state_dict()['weight'][pkey][0] = K_weights
            self.net2.state_dict()['weight'][0][pkey] = torch.from_numpy(self.kernel_dict[pkey]) ## 3x3 size kernel

    def forward(self, u):
        _,_,H,_ = u.shape
        u_split = self.net1(u)
        if(H == self.nnode_edge):
            g_pattern = self.global_pattern
        else:
            g_pattern = F.pad(self.global_pattern,(1,1,1,1),'constant', 0)
        u_split = u_split*g_pattern
        return self.net2(u_split)
    
    def convert_global_pattern(self, global_pattern_center):
        self.global_pattern = torch.zeros((1, self.n_channel, self.nnode_edge, self.nnode_edge))
        for pkey in self.kernel_dict:
            self.global_pattern[0,pkey,:,:] = torch.from_numpy(global_pattern_center[pkey]).reshape(self.nnode_edge, self.nnode_edge)

    def split_x(self, x):
        '''Split the field x based on the material phase'''
        x_split = self.net1(x)
        x_split = x_split*self.global_pattern
        return x_split

class FNet(nn.Module):
    def __init__(self, h):
        super(FNet, self).__init__()
        self.h = h
        self.net = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3, padding = 1, bias = False)
        f_weights_np = np.array([[h*h/36., h*h/9., h*h/36.],
                                [h*h/9., 4.*h*h/9., h*h/9.],
                                [h*h/36., h*h/9., h*h/36.]], dtype=np.float32).reshape(1,1,3,3)
        f_weights = torch.from_numpy(f_weights_np)
        self.net.weight = nn.Parameter(f_weights)

    def forward(self, x):
        return self.net(x)

