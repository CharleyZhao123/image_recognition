import torch
import numpy as np
import torch.nn.functional
from utils import reid_metric, fuse_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Model evaluator for val set
class R1_mAP(object):
    def __init__(self, feat_norm='yes', metric='euclidean', period=None):
        super(R1_mAP, self).__init__()
        assert metric in [
            'cosine', 'euclidean'
        ], 'reid_metric error: method: {} is not defined!'.format(metric)
        self.feat_norm = feat_norm
        self.metric = metric
        self.period = period
        self.gallery_baggage_ids = []
        self.probe_baggage_ids = []
        self.gallery_features = []
        self.probe_features = []

    def reset(self):
        self.period = None
        self.gallery_baggage_ids = []
        self.probe_baggage_ids = []
        self.gallery_features = []
        self.probe_features = []

    def update(self, output, feature_type):  # called once for each batch
        assert self.period in ['val', 'inference'], \
            'Evaluator Error: period: {} is not defined!'.format(self.period)
        assert feature_type in [
            'gallery', 'probe'
        ], 'Feature type error: {} is not defined!'.format(feature_type)

        if feature_type is 'gallery':
            gallery_feature, gallery_baggage_id = output
            self.gallery_features.append(gallery_feature)
            self.gallery_baggage_ids.extend(np.asarray(gallery_baggage_id))
        else:
            if self.period is 'val':
                probe_feature, probe_baggage_id = output
                self.probe_features.append(probe_feature)
                self.probe_baggage_ids.extend(np.asarray(probe_baggage_id))
            else:
                probe_feature = output
                self.probe_features.append(probe_feature)

    # called after each eval_period
    def compute(self, logger):
        probe_features = torch.cat(self.probe_features, dim=0)
        gallery_features = torch.cat(self.gallery_features, dim=0)
        if self.feat_norm == 'yes':
            logger.info("The test feature is normalized")
            probe_features = torch.nn.functional.normalize(
                probe_features, dim=1, p=2)  # along channel
            gallery_features = torch.nn.functional.normalize(
                gallery_features, dim=1, p=2)  # along channel

        probe_baggage_ids = np.asarray(self.probe_baggage_ids)
        gallery_baggage_ids = np.asarray(self.gallery_baggage_ids)

        if self.metric == 'euclidean':
            logger.info('=> Computing DistMat with euclidean distance')
        else:
            logger.info('=> Computing DistMat with cosine similarity')

        distance_mat = reid_metric(probe_features,
                                   gallery_features,
                                   mode=self.metric)

        eval_func(distance_mat,
                  probe_baggage_ids,
                  gallery_baggage_ids,
                  logger=logger)


# todo：把 distance_mat 和 probe_baggage_ids, gallery_baggage_ids 本地化
def eval_func(distance_mat,
              probe_baggage_ids,
              gallery_baggage_ids,
              max_rank=100,
              logger=None):
    """
        Evaluation with cmc curve and mAP.
    """
    unique_gallery_baggage_ids = np.unique(gallery_baggage_ids)
    num_probe, num_gallery = distance_mat.shape

    if num_gallery < max_rank:
        max_rank = num_gallery
        logger.info(
            "Note: number of gallery samples is quite small, got {}".format(
                num_gallery))

    merge_confidence = None
    modes = ['origin', 'min', 'max', 'mean']
    for mode in modes:
        logger.info("------ Reid evaluate in {} mode ------".format(mode))
        if mode is 'origin':
            indices = np.argsort(distance_mat, axis=1)
            matches = (gallery_baggage_ids[indices] ==
                       probe_baggage_ids[:, np.newaxis]).astype(np.int32)
        else:
            if merge_confidence is None:
                merge_confidence = []
                for i in range(len(probe_baggage_ids)):
                    merge_gallery_confidence = []
                    for j in unique_gallery_baggage_ids:
                        gallery_confidence_indices = np.where(
                            gallery_baggage_ids == j)
                        gallery_confidence = distance_mat[i][
                            gallery_confidence_indices]
                        merge_gallery_confidence.append(gallery_confidence)
                    merge_confidence.append(
                        merge_gallery_confidence
                    )  # have the same sequence with probe_name

            fuse = fuse_score(merge_confidence, mode=mode)
            num_probe, num_gallery = fuse.shape
            fuse_indices = np.argsort(fuse, axis=1)
            matches = (unique_gallery_baggage_ids[fuse_indices] ==
                       probe_baggage_ids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each probe
        all_cmc = []
        all_AP = []
        num_valid_probe = 0.  # number of valid probe

        for probe_index in range(num_probe):
            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[probe_index]

            if not np.any(orig_cmc):
                # this condition is true when probe identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_probe += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_probe > 0, "Error: all probe identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_probe
        mAP = np.mean(all_AP)

        if logger is not None:
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 3, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(
                    r, all_cmc[r - 1]))
