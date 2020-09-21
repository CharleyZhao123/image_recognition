import torch
import numpy as np


def cosine_distance(probe_feature, gallery_feature):
    epsilon = 0.00001
    cosine_distance_mat = probe_feature.mm(gallery_feature.t())
    probe_feature_norm = torch.norm(probe_feature, p=2, dim=1,
                                    keepdim=True)  # mx1
    gallery_feature_norm = torch.norm(gallery_feature,
                                      p=2,
                                      dim=1,
                                      keepdim=True)  # nx1
    probe_gallery_norm_dot = probe_feature_norm.mm(gallery_feature_norm.t())

    cosine_distance_mat = cosine_distance_mat.mul(
        1 / probe_gallery_norm_dot).cpu().numpy()
    cosine_distance_mat = np.clip(cosine_distance_mat, -1 + epsilon,
                                  1 - epsilon)
    cosine_distance_mat = np.arccos(cosine_distance_mat)
    return cosine_distance_mat


def euclidean_distance(probe_feature, gallery_feature):
    m = probe_feature.shape[0]
    n = gallery_feature.shape[0]
    euclidean_distance_mat = torch.pow(probe_feature, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                             torch.pow(gallery_feature, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    euclidean_distance_mat.addmm_(1, -2, probe_feature, gallery_feature.t())
    return euclidean_distance_mat.cpu().numpy()


embedding_metrics = {
    'cosine': cosine_distance,
    'euclidean': euclidean_distance
}


def reid_metric(probe_feature, gallery_feature, mode):
    metric = embedding_metrics[mode]
    distance_mat = metric(probe_feature, gallery_feature)
    return distance_mat
