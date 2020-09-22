# Image Recognition (Classification)


## 配置

### 环境

```
pytorch>1.1
...
```

### 数据

目前是简单的5分类任务，数据取自MiniImageNet，每类600张图像，其中训练、验证、测试集分别为360、120、120张。

数据地址：20900服务器/space1/zhaoqing/dataset/fsl/mini-imagenet下的/image文件夹是图像数据，/mini_ic里是划分的.csv文件。

## 训练&验证

运行脚本run_train.sh

结果：

```yaml
2020-09-22 15:12:35,354 image_classification.train INFO: Epoch[1/100] Iteration[1/29] Train_Loss: 0.804, Train_Acc: 0.297, Base Lr: 3.27e-05
2020-09-22 15:12:35,667 image_classification.train INFO: Epoch[1/100] Iteration[2/29] Train_Loss: 0.801, Train_Acc: 0.352, Base Lr: 3.27e-05
2020-09-22 15:12:36,038 image_classification.train INFO: Epoch[1/100] Iteration[3/29] Train_Loss: 0.799, Train_Acc: 0.380, Base Lr: 3.27e-05
2020-09-22 15:12:36,405 image_classification.train INFO: Epoch[1/100] Iteration[4/29] Train_Loss: 0.797, Train_Acc: 0.422, Base Lr: 3.27e-05
2020-09-22 15:12:36,743 image_classification.train INFO: Epoch[1/100] Iteration[5/29] Train_Loss: 0.794, Train_Acc: 0.506, Base Lr: 3.27e-05
2020-09-22 15:12:37,047 image_classification.train INFO: Epoch[1/100] Iteration[6/29] Train_Loss: 0.791, Train_Acc: 0.568, Base Lr: 3.27e-05
2020-09-22 15:12:37,360 image_classification.train INFO: Epoch[1/100] Iteration[7/29] Train_Loss: 0.788, Train_Acc: 0.589, Base Lr: 3.27e-05
2020-09-22 15:12:37,688 image_classification.train INFO: Epoch[1/100] Iteration[8/29] Train_Loss: 0.785, Train_Acc: 0.619, Base Lr: 3.27e-05
2020-09-22 15:12:38,062 image_classification.train INFO: Epoch[1/100] Iteration[9/29] Train_Loss: 0.782, Train_Acc: 0.646, Base Lr: 3.27e-05
2020-09-22 15:12:38,386 image_classification.train INFO: Epoch[1/100] Iteration[10/29] Train_Loss: 0.779, Train_Acc: 0.666, Base Lr: 3.27e-05
2020-09-22 15:12:38,695 image_classification.train INFO: Epoch[1/100] Iteration[11/29] Train_Loss: 0.776, Train_Acc: 0.679, Base Lr: 3.27e-05
2020-09-22 15:12:39,008 image_classification.train INFO: Epoch[1/100] Iteration[12/29] Train_Loss: 0.773, Train_Acc: 0.682, Base Lr: 3.27e-05
2020-09-22 15:12:39,344 image_classification.train INFO: Epoch[1/100] Iteration[13/29] Train_Loss: 0.770, Train_Acc: 0.696, Base Lr: 3.27e-05
2020-09-22 15:12:39,666 image_classification.train INFO: Epoch[1/100] Iteration[14/29] Train_Loss: 0.766, Train_Acc: 0.703, Base Lr: 3.27e-05
2020-09-22 15:12:39,978 image_classification.train INFO: Epoch[1/100] Iteration[15/29] Train_Loss: 0.763, Train_Acc: 0.711, Base Lr: 3.27e-05
2020-09-22 15:12:40,286 image_classification.train INFO: Epoch[1/100] Iteration[16/29] Train_Loss: 0.759, Train_Acc: 0.718, Base Lr: 3.27e-05
2020-09-22 15:12:40,624 image_classification.train INFO: Epoch[1/100] Iteration[17/29] Train_Loss: 0.755, Train_Acc: 0.728, Base Lr: 3.27e-05
2020-09-22 15:12:40,987 image_classification.train INFO: Epoch[1/100] Iteration[18/29] Train_Loss: 0.751, Train_Acc: 0.738, Base Lr: 3.27e-05
2020-09-22 15:12:41,307 image_classification.train INFO: Epoch[1/100] Iteration[19/29] Train_Loss: 0.747, Train_Acc: 0.748, Base Lr: 3.27e-05
2020-09-22 15:12:41,621 image_classification.train INFO: Epoch[1/100] Iteration[20/29] Train_Loss: 0.743, Train_Acc: 0.755, Base Lr: 3.27e-05
2020-09-22 15:12:41,945 image_classification.train INFO: Epoch[1/100] Iteration[21/29] Train_Loss: 0.739, Train_Acc: 0.764, Base Lr: 3.27e-05
2020-09-22 15:12:42,314 image_classification.train INFO: Epoch[1/100] Iteration[22/29] Train_Loss: 0.734, Train_Acc: 0.772, Base Lr: 3.27e-05
2020-09-22 15:12:42,661 image_classification.train INFO: Epoch[1/100] Iteration[23/29] Train_Loss: 0.731, Train_Acc: 0.779, Base Lr: 3.27e-05
2020-09-22 15:12:42,997 image_classification.train INFO: Epoch[1/100] Iteration[24/29] Train_Loss: 0.726, Train_Acc: 0.786, Base Lr: 3.27e-05
2020-09-22 15:12:43,306 image_classification.train INFO: Epoch[1/100] Iteration[25/29] Train_Loss: 0.722, Train_Acc: 0.792, Base Lr: 3.27e-05
2020-09-22 15:12:43,638 image_classification.train INFO: Epoch[1/100] Iteration[26/29] Train_Loss: 0.716, Train_Acc: 0.800, Base Lr: 3.27e-05
2020-09-22 15:12:44,009 image_classification.train INFO: Epoch[1/100] Iteration[27/29] Train_Loss: 0.712, Train_Acc: 0.807, Base Lr: 3.27e-05
2020-09-22 15:12:44,323 image_classification.train INFO: Epoch[1/100] Iteration[28/29] Train_Loss: 0.707, Train_Acc: 0.814, Base Lr: 3.27e-05
2020-09-22 15:12:46,339 image_classification.train INFO: Epoch[1/100] Iteration[29/29] Train_Loss: 0.706, Train_Acc: 0.807, Base Lr: 3.27e-05
2020-09-22 15:12:46,675 image_classification.train INFO: Epoch 1 done. Time per batch: 0.549[s] Speed: 116.7[samples/s]
2020-09-22 15:12:49,118 image_classification.train INFO: Epoch[1/100] Iteration[1/10] Val_Loss: 0.475, Val_Acc: 0.969
2020-09-22 15:12:49,222 image_classification.train INFO: Epoch[1/100] Iteration[2/10] Val_Loss: 0.460, Val_Acc: 0.977
2020-09-22 15:12:49,332 image_classification.train INFO: Epoch[1/100] Iteration[3/10] Val_Loss: 0.458, Val_Acc: 0.984
2020-09-22 15:12:49,443 image_classification.train INFO: Epoch[1/100] Iteration[4/10] Val_Loss: 0.463, Val_Acc: 0.984
2020-09-22 15:12:49,554 image_classification.train INFO: Epoch[1/100] Iteration[5/10] Val_Loss: 0.497, Val_Acc: 0.969
2020-09-22 15:12:49,661 image_classification.train INFO: Epoch[1/100] Iteration[6/10] Val_Loss: 0.513, Val_Acc: 0.964
2020-09-22 15:12:49,796 image_classification.train INFO: Epoch[1/100] Iteration[7/10] Val_Loss: 0.514, Val_Acc: 0.964
2020-09-22 15:12:49,916 image_classification.train INFO: Epoch[1/100] Iteration[8/10] Val_Loss: 0.498, Val_Acc: 0.965
2020-09-22 15:12:50,279 image_classification.train INFO: Epoch[1/100] Iteration[9/10] Val_Loss: 0.471, Val_Acc: 0.969
2020-09-22 15:12:50,833 image_classification.train INFO: Epoch[1/100] Iteration[10/10] Val_Loss: 0.463, Val_Acc: 0.968
```

## 测试

未完成
