import torch
import torch.optim


def build_optimizer(
        cfg,
        model,
):
    metric = cfg.MODEL.LOSS_TYPE
    base_learning_rate = cfg.SOLVER.BASE_LR  # 3e-4
    learning_rate_weight_decay = cfg.SOLVER.WEIGHT_DECAY  # 0.0005
    learning_rate_bias = cfg.SOLVER.BIAS_LR_FACTOR  # 2
    model_optimizer = cfg.SOLVER.OPTIMIZER_NAME  # 'Adam'
    learning_rate_center = cfg.SOLVER.CENTER_LR  # 0.5

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_learning_rate
        weight_decay = learning_rate_weight_decay
        if "bias" in key:
            lr = base_learning_rate * learning_rate_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if model_optimizer == 'SGD':
        optimizer = getattr(torch.optim, model_optimizer)(params, momentum=0.9)
    else:
        optimizer = getattr(torch.optim, model_optimizer)(params)

    return optimizer


if __name__ == '__main__':
    from model.resnet50 import ResNet50

    net = ResNet50(last_stride=2)
    resnet50_optimizer = build_optimizer(net)
