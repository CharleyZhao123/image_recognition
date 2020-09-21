import torch.nn.functional as f
from loss.softmax_with_label_smoothing import CrossEntropyLabelSmooth


# def build_loss(score, feat, target):
def build_loss(cfg, num_classes):
    loss_metric = cfg.MODEL.LOSS_TYPE  # [’default', 'label_smoothing']
    id_loss_weight = cfg.SOLVER.ID_LOSS_WEIGHT  # 1

    label_smoothing = CrossEntropyLabelSmooth(
        num_classes=num_classes)  # softmax_with_label_smoothing

    def loss_func(score, target):
        loss = id_loss_weight * f.cross_entropy(score, target)  # softmax loss

        if 'label_smoothing' in loss_metric:
            loss = id_loss_weight * label_smoothing(
                score, target)  # cover softmax loss
        return loss

    return loss_func


if __name__ == '__main__':
    import torch.optim
    a, b = build_loss(3818)
    optimizer_center = torch.optim.SGD(b.parameters(), lr=0.5)
