from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import pdb
import numpy as np

class weightedContrastiveLoss(nn.Module):
    """Weighted contrastive loss.

    Reference:
    http://gitlab.hobot.cc/3D_Computer_Vision_Research_Group/3D_Optimal_Transport_Loss/blob/master/Step_04/model.py

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, lamb=10.0, same_margin=1.0):
        super(weightedContrastiveLoss, self).__init__()
        self.margin = margin
        #self.margin_pos = same_margin  # Modified by Sun 2019.1.21
        self.lamb = lamb
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs_batch1, inputs_batch2, targets_batch1, targets_batch2):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        # Similarity matrix
        simLabel = self.simLabelGeneration(targets_batch1, targets_batch2)
        #pdb.set_trace()

        # Caculating the ground matrix
        GM, expGM = self.calculateGroundMetricContrastive(inputs_batch1, inputs_batch2, simLabel)

        # Sinkhorn iteration construction
        T, T_flatten = self.sinkhornIter(expGM)

        # create loss
        loss = torch.sum(GM.mul(T_flatten))
        #self.loss_summary = tf.summary.scalar('wcontr_loss', loss)

        #return loss, torch.sum(T)
        return loss


    def simLabelGeneration(self, label1, label2):
        batch_size = label1.size(0)

        label_expand_batch1 = label1.view(batch_size, 1).repeat(1, batch_size) # batch_size * batch_size
        label_expand_batch2 = label2.view(batch_size, 1).repeat(batch_size, 1) # (batch_size * batch_size) * 1

        simLabel = torch.eq(label_expand_batch1.view(batch_size*batch_size, -1),
                            label_expand_batch2.view(batch_size*batch_size, -1)).float()
        simLabelMatrix = simLabel.view(batch_size, batch_size)

        #pdb.set_trace()
        return simLabelMatrix

    def calculateGroundMetricContrastive(self, batchFea1, batchFea2, labelMatrix):
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

        # Get ground metric cost for negative pair
        hinge_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), self.margin - groundMetric)
        #positive_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), groundMetric - self.margin_pos)

        # Work Version
        GM_positivePair = labelMatrix.mul(groundMetric)
        # Modified 2019.1.21
        #GM_positivePair = labelMatrix.mul(positive_groundMetric)
        GM_negativePair = (1 - labelMatrix).mul(hinge_groundMetric)
        GM = GM_negativePair + GM_positivePair

        GM_npy = GM.data.cpu().numpy()
        inverseGMPrenorm = np.max(GM_npy) - GM_npy
        inverseGM = inverseGMPrenorm / (np.max(inverseGMPrenorm) + 1e-12)
        
        # TODO Modified from -1 to -10
        #expGM = torch.exp(-10. * GM) # This is for optimizing "T"
        # Flatten the groundMetric as a vector
        GMFlatten = GM.view(-1)
        #pdb.set_trace()

        return GMFlatten, inverseGM

    def sinkhornIter(self, GM_numpy):
        batch_size = GM_numpy.shape[0]
        u0 = np.ones((batch_size, 1)) * (1 / batch_size)
        GM_numpy_ = GM_numpy.transpose()

        epsilon = 1e-12
        u = np.ones((batch_size, 1))
        v = np.ones((batch_size, 1))

        for ind in range(200):
            v_iter = u0 * 1./(np.matmul(np.exp(-self.lamb * GM_numpy_), u) + epsilon)
            v = v_iter.copy()
            u_iter = u0 * 1./(np.matmul(np.exp(-self.lamb * GM_numpy), v) + epsilon)
            u = u_iter.copy()

        T_numpy = np.matmul(np.matmul(np.diag(np.reshape(u, [-1])), np.exp(-self.lamb * GM_numpy)), np.diag(np.reshape(v, [-1])))
        T_flatten_numpy = np.reshape(T_numpy, [-1])
        T = torch.autograd.Variable(torch.from_numpy(T_numpy)).float().cuda()
        T_flatten = torch.autograd.Variable(torch.from_numpy(T_flatten_numpy)).float().cuda()
        #pdb.set_trace()
        return T, T_flatten