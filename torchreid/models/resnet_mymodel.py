# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import os
import random
from torch.autograd import Variable
import numpy as np
import pdb
import pickle
#import gensim

__all__ = ['ResNet50']


class GlobalVision(nn.Module):
    def __init__(self, freeze_resnet=False):
        super(GlobalVision, self).__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if freeze_resnet:
            for params in resnet50.parameters():
                params.requires_grad = False
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        x_conv_global = self.base(x) #[32, 2048, 8, 4]
        f_global = F.avg_pool2d(x_conv_global, x_conv_global.size()[2:]).squeeze() #[32, 2048]

        return f_global


class Language(nn.Module):
    def __init__(self):
        super(Language, self).__init__()

        VOCAB_SIZE = 6207
        self.word_emb_dim = 512
        self.hidden_dim = 512

        self.emb = nn.Embedding(VOCAB_SIZE, self.word_emb_dim)

        self.lstm = nn.LSTM(self.word_emb_dim, self.hidden_dim, batch_first=True)
        self.lstm_top = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.emb_g= nn.Linear(2048+self.hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

        self.emb_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_x = nn.Linear(2048, 2048)

        self.maxpooling = nn.MaxPool2d(kernel_size=(30, 1))


    def init_hidden(self, emb):
        if self.training:
            return (Variable(torch.zeros(1, emb.size(0), self.hidden_dim).cuda()),
                    Variable(torch.FloatTensor(1, emb.size(0), self.hidden_dim).uniform_(-0.1,0.1).cuda())
                    )
        else:
            return (torch.zeros(1, emb.size(0), self.hidden_dim).cuda(),
                    #torch.zeros(1, emb.size(0), self.hidden_dim).cuda()
                    torch.FloatTensor(1, emb.size(0), self.hidden_dim).uniform_(-0.1,0.1).cuda())

    def _sample_gumbel(self, shape, out=None):
        eps = 1e-10
        U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        noise = torch.log(U + eps) - torch.log(1 - U + eps)
        return noise

    def _gumbel_sigmoid_sample(self, logits, tau=0.8):
        gumbel_noise = Variable(self._sample_gumbel(logits.size(), out=logits.data.new()))
        y = (logits + gumbel_noise) / tau
        return F.sigmoid(y)

    def gumbel_sigmoid(self, logits, tau=0.8, hard=True):
        y_soft = self._gumbel_sigmoid_sample(logits, tau=tau)
        if hard: # Hard gate
            t = Variable(torch.Tensor([0.5]).cuda())# threshold
            y_hard = (y_soft >= t).float() * 1
            y = y_hard - y_soft.detach() + y_soft
        else: # Soft gate
            y = y_soft
        return y, y_soft


    def forward(self, x, y):
        y = y.type(torch.LongTensor).cuda()

        # word embedding
        y_emb = self.emb(y)  # [B, 30, 512]

        # initialize h and c of LSTMs
        h0, c0 = self.init_hidden(y_emb) # [1,B,512]
        h_top_0, c_top_0 = self.init_hidden(y_emb)  # [1,B,512]

        OutputTop=[]
        HTop=[]
        for (i,yt) in enumerate(y_emb.permute(1,0,2)):
            # Bottom LSTM
            output, (h0, c0) = self.lstm(yt[:,None,:], (h0, c0))

            # Gate features combining h of bottom LSTM and image feature
            h_emb = self.emb_h(h0.squeeze())
            x_emb = self.emb_x(x)
            gate_features = torch.cat([h_emb, x_emb], dim=1)
            g_features = self.emb_g(gate_features) #[32, 1]

            # gate
            gate, y_soft = self.gumbel_sigmoid(g_features, tau=0.3, hard=True)

            # Top LSTM
            h_features = gate * h0.squeeze()
            output_top, (h_top_0_b, c_top_0_b) = self.lstm_top(h_features[:, None, :], (h_top_0, c_top_0))
            c_top_0 = gate * c_top_0_b + (1. - gate) * c_top_0
            h_top_0 = gate * h_top_0_b + (1. - gate) * h_top_0


            OutputTop.append(output_top)
            HTop.append(h_top_0.permute(1, 0, 2))

        # average pooling
        h_cat = torch.cat(HTop, 1)
        h_mean = torch.mean(h_cat, dim=1).cuda()
        pred = h_mean

        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)



class ResNet50(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__()

        self.global_vision = GlobalVision(freeze_resnet=False)
        # Load pretrained model and freeze
        #load_pre_model(model=self.global_vision, checkpoint_path='/data/jun.xu/log/models201905/checkpoint_ep175_GlobalVision.pth.tar',
        #               freeze=True)

        self.language = Language()
        #load_pre_model(model=self.language, checkpoint_path='/data/jun.xu/log/models201905/checkpoint_ep615_LanguageImg.pth.tar',
        #               freeze=True)

        self.classifier = nn.Linear(256+2048, num_classes)

        self.emb_language = nn.Linear(512, 256)
        self.emb_vision = nn.Linear(2048, 2048)


    def forward(self, x, y):
        # Extract global features
        f_global = self.global_vision(x)

        # Extract language features by LSTM
        f_language = self.language(f_global,y)

        # Global Vision and Language
        f_emb_language = self.emb_language(f_language)
        f_emb_global = self.emb_vision(f_global)

        f_final = torch.cat([f_emb_language,f_emb_global], dim=1)
        cls_result = self.classifier(f_final)

        if not self.training:
            return f_final

        return cls_result, f_final


# Load pretrained model
def load_pre_model(model, checkpoint_path, freeze=False):
    #for params in model.base.parameters():
    #    print(params)
    checkpoint = torch.load(checkpoint_path)
    pretrain_dict = checkpoint['state_dict']  # eg. 'global_vision.base.0.weight'
    model_dict = model.state_dict()  # eg. 'base.0.weight'
    pretrain_dict = {k[k.find('.') + 1:]: v for k, v in pretrain_dict.items()} # 去掉'global_vision.'
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                     k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Loaded pretrained weights from '{}'".format(checkpoint_path))
    #print('\n\n\n\nttttt')
    #for params in model.base.parameters():
    #    print(params)

    if freeze:
        for params in model.parameters():
            params.requires_grad = False


