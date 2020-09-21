import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularize.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        :param num_classes (int): number of classes.
        :param epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            :param inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            :param targets: ground truth labels with shape (num_classes)
        """
        log_probability = self.log_softmax(inputs)
        targets = torch.zeros(log_probability.size()).scatter_(
            1,
            targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probability).mean(0).sum()
        return loss
