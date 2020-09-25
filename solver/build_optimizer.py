import torch
import torch.optim

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

def build_optimizer(
        cfg,
        model,
):
    metric = cfg.MODEL.LOSS_TYPE
    base_learning_rate = cfg.SOLVER.BASE_LR  # 3e-4
    learning_rate_weight_decay = cfg.SOLVER.WEIGHT_DECAY  # 0.0005
    learning_rate_bias = cfg.SOLVER.BIAS_LR_FACTOR  # 2
    model_optimizer = cfg.SOLVER.OPTIMIZER_NAME  # 'Adam'

    model.apply(set_bn_eval) # freeze bn 
    params = []
    for key, value in model.named_parameters():

        # freeze model
        # flag = 'classifier' in key or 'layer4.2' in key
        flag = 'classifier' in key or 'layer4' in key
        if not value.requires_grad or not flag:
            continue
        # print(key)

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
