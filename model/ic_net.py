import torch
from torch import nn
from model.resnet50 import ResNet50, Bottleneck, ResNet101
from model.resnet_ibn_a import resnet50_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ClassificationNet(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path,
                 model_name, pretrained_choice):
        super(ClassificationNet, self).__init__()
        if model_name == 'resnet50':
            self.backbone = ResNet50(last_stride=last_stride, block=Bottleneck)
        if model_name == 'resnet101':
            self.backbone = ResNet101(last_stride=last_stride,
                                      block=Bottleneck)
        elif model_name == 'resnet50_ibn_a':
            self.backbone = resnet50_ibn_a(last_stride)

        if pretrained_choice == 'imagenet':
            self.backbone.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # classifier tree method

        # self.classifier_list = nn.Sequential()
        # for i in range(0,self.num_classes-1):
        #     classifier = nn.Linear(self.in_planes, 1, bias=False)
        #     classifier.apply(weights_init_classifier)
        #     self.classifier_list.add_module('classifier'+str(i),classifier)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.gap(self.backbone(x))  # (b, 2048, 1, 1)
        feat = feat.view(feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.training:
            cls_score = self.classifier(feat)

            # classifier tree method

            # score = []
            # out_score = []
            # sum_value = 0
            # for i in range(0,self.num_classes-1):
            #     cls_score = self.classifier_list[i](feat)
            #     # print(cls_score.device)
            #     score.append(self.sigmoid(cls_score))
            # for i in range(0,self.num_classes):
            #     neg_rate = 1
            #     for j in range(0,i):
            #         neg_rate *= (1-score[j])
            #     if i<self.num_classes-1:
            #         out_score.append(neg_rate * score[i])
            #         sum_value += neg_rate * score[i]
            #     else:
            #         last_value = 1-sum_value
            #         last_value[last_value<0] = 0
            #         out_score.append(last_value)
            # # print(len(out_score))
            # cls_score = torch.cat(out_score,axis=1)
            # # print(cls_score.shape)

            return cls_score
        else:
            cls_score = self.classifier(feat)

            # classifier tree method

            # score = []
            # out_score = []
            # sum_value = 0
            # for i in range(0,self.num_classes-1):
            #     cls_score = self.classifier_list[i](feat)
            #     # print(cls_score.device)
            #     score.append(self.sigmoid(cls_score))
            # for i in range(0,self.num_classes):
            #     neg_rate = 1
            #     for j in range(0,i):
            #         neg_rate *= (1-score[j])
            #     if i<self.num_classes-1:
            #         out_score.append(neg_rate * score[i])
            #         sum_value += neg_rate * score[i]
            #     else:
            #         last_value = 1-sum_value
            #         last_value[last_value<0] = 0
            #         out_score.append(last_value)
            # # print(len(out_score))
            # cls_score = torch.cat(out_score,axis=1)
            # # print(cls_score.shape)

            return cls_score

    # load pretrained model
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])


if __name__ == '__main__':
    icnet = ClassificationNet()
