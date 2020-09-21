import torch
import numpy as np
import logging
import torch.nn.functional
from utils import fuse_score, reid_metric, re_ranking


def generate_full_id(baggage_id):
    assert 0 < len(baggage_id) <= 4
    supplement = 4 - len(baggage_id)
    for i in range(supplement):
        baggage_id = '0' + baggage_id
    return baggage_id


def generate_merge_result(cfg,
                          probe_features,
                          gallery_features,
                          probe_image_names,
                          gallery_baggage_ids,
                          metric='cosine',
                          result_path=None):
    assert metric in ['euclidean', 'cosine'], 'metric error!'

    probe_features = torch.cat(probe_features, dim=0)
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_baggage_ids = np.asarray(gallery_baggage_ids)
    unique_gallery_baggage_ids = np.unique(gallery_baggage_ids)

    if cfg.TEST.FEAT_NORM == 'yes':
        probe_features = torch.nn.functional.normalize(probe_features,
                                                       dim=1,
                                                       p=2)  # along channel
        gallery_features = torch.nn.functional.normalize(gallery_features,
                                                         dim=1,
                                                         p=2)  # along channel

    #re-ranking
    if cfg.TEST.RE_RANKING == 'yes':
        logger = logging.getLogger('{}.inference'.format(cfg.PROJECT.NAME))
        logger.info("Enter reranking")
        distance_mat = re_ranking(probe_features,
                                  gallery_features,
                                  mode=metric)
    else:
        distance_mat = reid_metric(probe_features,
                                   gallery_features,
                                   mode=metric)

    int_probe = []
    for name in probe_image_names:
        int_probe.append(int(name))
    int_probe = np.asarray(int_probe)
    int_probe_sort_indices = np.argsort(int_probe)

    all_confidence = []
    for i, probe_image_name in enumerate(probe_image_names):
        index = int_probe_sort_indices[i]
        merge_gallery_confidence = []
        for j in unique_gallery_baggage_ids:
            gallery_confidence_indices = np.where(gallery_baggage_ids == j)
            gallery_confidence = distance_mat[index][
                gallery_confidence_indices]
            merge_gallery_confidence.append(gallery_confidence)
        all_confidence.append(
            merge_gallery_confidence)  # have the same sequence with probe_name

    max_fuse = fuse_score(all_confidence, mode='mean')
    # min_fuse = fuse_score(all_confidence, mode='min')
    # max_fuse = fuse_score(all_confidence, mode='max')

    max_fuse_rank = -np.sort(-max_fuse, axis=1)
    max_fuse_indices = np.argsort(-max_fuse, axis=1)
    gallery_baggage_ids_rank = unique_gallery_baggage_ids[max_fuse_indices]

    probe_image_names_sort = []
    for i in int_probe_sort_indices:
        probe_image_names_sort.append(probe_image_names[i])
    final_rank_sort = max_fuse_rank[int_probe_sort_indices]

    if result_path is None:
        result_path = '/space1/home/chenyanxian/RemoteProject/MVB_Reid/045_bag_result_test_final_mean.csv'

    inference_num_probe = len(probe_image_names_sort)
    with open(result_path, 'w') as f:
        for i in range(inference_num_probe):
            f.write(probe_image_names_sort[i] + ',')  # row name
            for j in range(gallery_baggage_ids_rank.shape[1]):
                str_gallery_baggage_id = gallery_baggage_ids_rank[i][j].astype(
                    str)
                gallery_baggage_full_id = generate_full_id(
                    str_gallery_baggage_id)
                str_gallery_confidence = str(
                    final_rank_sort[i][j].item())[:6]  # column value
                f.write(gallery_baggage_full_id + ',')
                if j < gallery_baggage_ids_rank.shape[1] - 1:
                    f.write(str_gallery_confidence + ',')
                else:
                    f.write(str_gallery_confidence)
            f.write('\n')

    print('Success generate reid merge results')


if __name__ == '__main__':
    # import torch.cuda
    # from model import build_model
    # from data import build_dataloader
    #
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model_path = '/space1/home/chenyanxian/RemoteProject/MVB_Reid/MVB_Reid_experiment_1_120.pth'
    #
    # model = build_model()
    # model.load_param(model_path)
    # model.to(device)
    # model.eval()
    #
    # # dataloader for inference
    # inference_period = 'inference'
    # inference_gallery_loader = build_dataloader(period=inference_period, loader_type='gallery')
    # inference_probe_loader, inference_num_probe = build_dataloader(period=inference_period, loader_type='probe')
    #
    # probe_features = []
    # probe_image_names = []
    # gallery_features = []
    # gallery_baggage_ids = []
    #
    # for iteration, (img, baggage_id) in enumerate(inference_gallery_loader):
    #     with torch.no_grad():
    #         img = img.to(device)
    #         inference_gallery_feature = model(img)
    #         gallery_features.append(inference_gallery_feature)
    #         gallery_baggage_ids.extend(np.asarray(baggage_id))
    #
    # for iteration, (img, _, image_name) in enumerate(inference_probe_loader):
    #     with torch.no_grad():
    #         img = img.to(device)
    #         inference_probe_feature = model(img)
    #         probe_features.append(inference_probe_feature)
    #         probe_image_names.extend(image_name)
    #
    # generate_merge_result(probe_features, gallery_features, probe_image_names, gallery_baggage_ids, metric='cosine')
    pass
