# 目录

# 模型名称

> 模型: ConvLSTM
>
> 论文链接: https://arxiv.org/pdf/1506.04214.pdf
>
> 简介:
>
> ​	模型ConvLSTM作者通过实验**证明了ConvLSTM在获取时空关系上比LSTM有更好的效果。**ConvLSTM不仅可以预测天气，还能够解决其他时空序列的预测问题。比如视频分类，动作识别等。此次数据集为[Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)

## 模型架构

> ![image-20220727091656901](https://user-images.githubusercontent.com/70456146/181144611-67b2a5ef-6b7e-4310-aaa1-fd29308cd054.png)

## 数据集

> 训练数据集: [The MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
>
> 验证精度数据集: [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
>
> 下载数据集请通过目录下的脚本下载
>
> ```
> source ./download.sh
> ```

## 环境

> Environment(Ascend/GPU/CPU): GPU-GTX3090(24G)
> Software Environment:
> – MindSpore version : 1.7.0
> – Python version : 3.8.13
> – OS platform and distribution : Ubuntu 16.04
> – CUDA version : 11.0

## 快速入门

> 训练命令:
>
> ```
> python train.py --batch_size 24 -checkpoints 'checkpoint_66_0.000961.ckpt' -epochs 500
> ```
>
> 评估命令:
>
> ```
> python eval.py --batch_size 24 -checkpoints 'checkpoint_66_0.000961.ckpt'
> ```

## 训练过程

> 训练过程采用MNIST手写数字数据库，其中有60,000个示例的训练集和10,000个示例的测验集。它是MNIST的子集。数字已被归一化并以固定大小的图像为中心。训练及测验过程中通过动态生成视频数据来进行训练。特别需要注意，训练过程中生成的数字数量为3，相较于评估中的2个数字更多。

### 训练

通过以下指令启动训练。保存的参数模型将存于当前目录save_model中

```
python train.py --batch_size 24 -checkpoints 'checkpoint_66_0.000961.ckpt' -epochs 500
```

## 评估

### 评估过程

> 评估过程采用[Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)作为测试集。

### 评估结果

> 通过以下指令启动评估。

```log
python eval.py --batch_size 24 -checkpoints 'checkpoint_66_0.000961.ckpt'
```

## 性能

### 训练性能

| train_loss | valid_loss | SSIM     | MAE        | MSE       |
| ---------- | ---------- | -------- | ---------- | --------- |
| 0.000976   | 0.000961   | 0.777687 | 221.285598 | 94.498799 |

### 评估性能

| test_loss | SSIM     | MAE        | MSE       |
| --------- | -------- | ---------- | --------- |
| 0.000638  | 0.833904 | 156.482312 | 62.759463 |

## 随机情况说明

> 载入权重模型后继续训练会有较大精度浮动

## 参考模板

[ConvLSTM-PyTorch](https://github.com/jhhuang96/ConvLSTM-PyTorch)

## 贡献者

* [曾润佳](https://github.com/zRAINj) (广东工业大学)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。