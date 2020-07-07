from .RTdata_emb import Dictionary, Corpus
from .deeprt_metric import Pearson, Spearman

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


NUM_CLASSES = 10
NUM_ROUTING_ITERATIONS = 1


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, 
                 num_capsules, 
                 num_route_nodes, 
                 in_channels, 
                 out_channels, 
                 kernel_size=None, 
                 stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):

        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, 
                                                          num_route_nodes, 
                                                          in_channels, 
                                                          out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, 
                           out_channels, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


param_1D_rt = {'data': 'rt',
               'dim': 1,
               'stride': 1,
               'NUM_CLASSES': 1,
               }

param = param_1D_rt

if 1 == param['dim']:
    print('>> note: using seq mode.')


class CapsuleNet(nn.Module):
    def __init__(self, conv1_kernel, conv2_kernel, config, dictionary):
        super(CapsuleNet, self).__init__()

        emb_size = 20
        self.emb = nn.Embedding(len(dictionary), emb_size)

        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=256,
                               kernel_size=(emb_size, conv1_kernel),  # param['conv1_kernel'], # (28, 9), # 9,
                               stride=1)
        ''''''
        self.bn1 = nn.BatchNorm2d(256) # Note: do we need this or not?
        self.conv2 = nn.Conv2d(in_channels=256, 
                               out_channels=256,  # 256
                               kernel_size=(1, conv1_kernel),  # (28, 9), # 9,
                               stride=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.primary_capsules = CapsuleLayer(num_capsules=8,  # 8
                                             num_route_nodes=-1, 
                                             in_channels=256,  # 256
                                             out_channels=32,  # 32
                                             kernel_size=(1, conv2_kernel),  # param['pri_caps_kernel'], # (1, 9), # 9,
                                             stride=param['stride'])  # 1) # 2)

        self.digit_capsules = CapsuleLayer(num_capsules=param['NUM_CLASSES'],  # 1, #NUM_CLASSES, # DeepRT
                                           num_route_nodes=32 * 1 * (config['Max_Len'] - conv1_kernel*2 + 2 - conv2_kernel + 1),  # param['digit_caps_nodes'], # 32 * 1 * 12, # 32 * 6 * 6,
                                           in_channels=8,  # 8
                                           out_channels=16)  # max_length-conv1_kernel + 1) # 16

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        # print('>>dim: input', x.shape) # [batch, 1, 28, 28]
        # print('>>dim: y', y) # [batch, 10] ~ [batch, NUM_CLASSES]

        x = self.emb(x) # [batch, len] -> [batch, len, dict]
        x = x.transpose(dim0=1, dim1=2) # -> [batch, dict, len]
        x = x[:, None, :, :] # -> [batch, 1, dict, len]

        # ^^^^^ pre-process x ^^^^^
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)      

        ''' try residue: not good! 
        residue = x.view(x.shape[0],-1)        
        residue = self.linear(residue).view(residue.shape[0],1,16)
        # another residue method
        residue = F.relu(self.conv_res(x), inplace=True)
        residue = residue.view(residue.shape[0],1,residue.shape[-1])
        '''

        # x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True) # improvement
        # x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        # print('>>dim: conv1', x.shape) # [batch, 256, 20, 20]
        x = self.primary_capsules(x)
        # print('>>dim: primary_capsules', x.shape) # [batch, 1152, 8] = [batch, 6*6*32, 8]
        # print('>>dim: unsqueezeed', self.digit_capsules(x).shape) # [10, batch, 1, 1, 16] ~ [num_caps, batch, ...]

        if 1 == param['dim']:
            x = self.digit_capsules(x).squeeze()[:, None, :]

            # [1, batch, 1, 1, 16] -> squeeze: [batch, 16]
        # print('>>dim: digit_capsules', x.shape) # [batch, 10, 16]

        # add dropout:
        # x = self.dropout(x)
        # x = self.linear(x)
        # x = F.sigmoid(x)
        
        # x = x + residue # try residue: not good!
        classes = (x ** 2).sum(dim=-1) ** 0.5
        # print('>>dim: classes', classes) # [batch, 10]
        if 2 == param['dim']:
            classes = F.softmax(classes) # DeepRT
        # print('>>dim: softmax', classes)

        if y is None: # Note: not do this during training. Here y is only used for reconstruction
            if 2 == param['dim']:
                # In all batches, get the most active capsule.
                # print('>>dim: reconstruction', classes) # [batch, 10]
                _, max_length_indices = classes.max(dim=1) 
                # give: [torch.FloatTensor of size batch] and [torch.FloatTensor of size batch]
                y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

                # generate a new y: [batch, 10] with each column having 1 in batch 0

        if 1 == param['dim']:
            return classes, x # Note here


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        # print('>>dim: labels', labels) # torch.cuda.FloatTensor of size batch x 1
        # print('>>dim: classes', classes) # [batch, 1]

        loss = ((labels - classes) ** 2).sum()/labels.shape[0]  # MSE # Note: here it must be sum()
        loss = loss ** 0.5  # RMSE

        # print('>>dim: loss', loss)
        return loss
