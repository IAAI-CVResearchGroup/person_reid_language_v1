from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import pdb
import numpy as np
import math
from torch.autograd import Variable

class KM_algorithm:
    def __init__(self, groundMetric):
        self.mp = groundMetric
        self.n = groundMetric.shape[0]
        self.link = np.zeros(self.n).astype(np.int16)
        self.lx = np.zeros(self.n)
        self.ly = np.zeros(self.n)
        self.sla = np.zeros(self.n)
        self.visx = np.zeros(self.n).astype(np.bool)
        self.visy = np.zeros(self.n).astype(np.bool)
        
    def DFS(self, x):
        self.visx[x] = True
        for y in range(self.n):
            if self.visy[y]:
                continue
            tmp = self.lx[x] + self.ly[y] - self.mp[x][y]
            if math.fabs(tmp) < 1e-5:
                self.visy[y] = True
                if self.link[y] == -1 or self.DFS(self.link[y]):
                    self.link[y] = x
                    return True
            elif self.sla[y] + 1e-5 > tmp: 
                self.sla[y] = tmp  
        return False
    
    def run(self):
        for index in range(self.n):
            self.link[index] = -1
            self.ly[index] = 0.0
            self.lx[index] = np.max(self.mp[index])
        
        for x in range(self.n):
            self.sla = np.zeros(self.n) + 1e10
            while True:
                self.visx = np.zeros(self.n).astype(np.bool)
                self.visy = np.zeros(self.n).astype(np.bool)
                if self.DFS(x): 
                    break
                d = 1e10
                for i in range(self.n):
                    if self.visy[i] == False:
                        d = min(d, self.sla[i])
                for i in range(self.n):
                    if self.visx[i]:
                        self.lx[i] -= d
                    if self.visy[i]:
                        self.ly[i] += d
                    else:
                        self.sla[i] -= d
        
        res = 0
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            if self.link[i] != -1:
                T[self.link[i]][i] = 1.0 / self.n
        return T
            

class weightedContrastiveLoss_KM(nn.Module):
    """Weighted contrastive loss.

    Reference:
    http://gitlab.hobot.cc/3D_Computer_Vision_Research_Group/3D_Optimal_Transport_Loss/blob/master/Step_04/model.py

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, lamb=10.0, same_margin = 0.0, mode = 'both', adaptive_margin=False):
        super(weightedContrastiveLoss_KM, self).__init__()
        self.margin = margin
        self.same_margin = same_margin
        self.lamb = lamb
        self.mode = mode
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.adaptive_margin = adaptive_margin
        #self.margin_var = torch.empty(1, 1).cuda()
        #nn.init.constant_(self.margin_var, 0)
        #self.margin_var = Variable(torch.zeros(1, 1), requires_grad=True).cuda()

    def forward(self, inputs_batch1, inputs_batch2, targets_batch1, targets_batch2, margin_var):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        # Similarity matrix
        simLabel = self.simLabelGeneration(targets_batch1, targets_batch2)
        #pdb.set_trace()

        # Caculating the ground matrix
        #margin_var = Variable(torch.zeros(1, 1)).cuda()

        GMFlatten, GM = self.calculateGroundMetricContrastive(inputs_batch1, inputs_batch2, simLabel, margin_var)

        # ====== Modified by Sun ======
        #batch_size = inputs_batch1.shape[0]
        #T = np.zeros((batch_size, batch_size))
        #GM_max = torch.max(GM, 1)
        #for ind in range(batch_size):
        #    T[ind, GM_max[1][ind]] = 1/batch_size
        # =============================
        KM = KM_algorithm(GM.data.cpu().numpy())
        T = KM.run()

        T_flatten = torch.autograd.Variable(torch.from_numpy(T.reshape([-1]))).float().cuda()
       
        # create loss
        loss = torch.sum(GMFlatten.mul(T_flatten))
        #pdb.set_trace()
        return loss, torch.sum(simLabel.mul(T_flatten.view(inputs_batch1.shape[0], inputs_batch1.shape[0]))) * inputs_batch1.shape[0], T

    def simLabelGeneration(self, label1, label2):
        batch_size = label1.size(0)

        label_expand_batch1 = label1.view(batch_size, 1).repeat(1, batch_size) # batch_size * batch_size
        label_expand_batch2 = label2.view(batch_size, 1).repeat(batch_size, 1) # (batch_size * batch_size) * 1

        simLabel = torch.eq(label_expand_batch1.view(batch_size*batch_size, -1),
                            label_expand_batch2.view(batch_size*batch_size, -1)).float()
        simLabelMatrix = simLabel.view(batch_size, batch_size)

        #pdb.set_trace()
        return simLabelMatrix

    def calculateGroundMetricContrastive(self, batchFea1, batchFea2, labelMatrix, margin_var):
        """
        calculate the ground metric between two batch of features
        """

        # Print the tensor shape for debug

        # print("Calculate the ground metrics between two batches of features")
        # print("batchFea1.get_shape().as_list() = {}".format(batchFea1.get_shape().as_list()))
        # print("batchFea2.get_shape().as_list() = {}".format(batchFea2.get_shape().as_list()))
        # print("labelMatrix.get_shape().as_list() = {}".format(labelMatrix.get_shape().as_list()))
        # ============== Euclidean Distance ==============
        batch_size = batchFea1.size(0)
        squareBatchFea1 = torch.sum(batchFea1.pow(2), 1)
        squareBatchFea1 = squareBatchFea1.view(batch_size, -1)

        squareBatchFea2 = torch.sum(batchFea2.pow(2), 1)
        squareBatchFea2 = squareBatchFea2.view(-1, batch_size)

        correlationTerm = batchFea1.mm(batchFea2.t())

        groundMetric = squareBatchFea1 - 2 * correlationTerm + squareBatchFea2
        # =================================================
        if self.adaptive_margin:
            # ====== Sun 2019.2.26 ======
            same_class_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), groundMetric - margin_var)
            hinge_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(),
                                           margin_var + self.margin - groundMetric)
        else:
            # Get ground metric cost for negative pair
            hinge_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), self.margin - groundMetric)
            same_class_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), groundMetric - self.same_margin)

        # ===========================

        GM_positivePair = labelMatrix.mul(same_class_groundMetric)
        GM_negativePair = (1 - labelMatrix).mul(hinge_groundMetric)
        if self.mode == 'both':
            GM = GM_negativePair + GM_positivePair
        elif self.mode == 'pos':
            GM = GM_positivePair + (1-labelMatrix).mul(-1000000000.0)
        else:
            GM = GM_negativePair + labelMatrix.mul(-1000000000.0)
        GMFlatten = GM.view(-1)
        return GMFlatten, GM
