import torch
import numpy as np
import torch.nn.functional


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
        # print(self.val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# todo：把 distance_mat 和 probe_baggage_ids, gallery_baggage_ids 本地化
def eval_func(distance_mat,
              probe_baggage_ids,
              gallery_baggage_ids,
              max_rank=100,
              logger=None):

        # compute cmc curve for each probe
        all_cmc = []
        all_AP = []

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

        if logger is not None:
            for r in [1, 3, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(
                    r, all_cmc[r - 1]))
