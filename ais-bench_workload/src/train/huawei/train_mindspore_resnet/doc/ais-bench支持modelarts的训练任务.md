## 基于ModelArts的集群训练Ais-Bench接入

### 背景
ModelArts华为云训练准备&入口，请参照[这里](https://support.huaweicloud.com/TensorFlowdevg-cann202training1/atlasmprtg_13_0046.html)。本文不做详细介绍。
### 整体流程

```sequence
本地侧->>ModelArts侧: 传递训练参数，拉起训练
ModelArts侧->>OBS侧: 请求下载训练代码
OBS侧->>ModelArts侧: 下载代码
ModelArts侧->>OBS侧: 请求下载数据集
OBS侧->>ModelArts侧: 下载数据集
ModelArts侧->>ModelArts侧: 执行训练
ModelArts侧->>OBS侧: 上传throughput/accuracy数据
ModelArts侧->>本地侧: 训练完成
本地侧->>OBS侧: 请求下载throughput/accuracy数据
OBS侧->>本地侧: 下载数据
本地侧->>Tester: 上报数据及运行结果
```

整体流程如上图所示，大致可分为3个步骤：

1. 用户在本地配置训练任务信息
2. 用户在本地拉起ais-bench-stubs二进制，整个训练过程和数据统计过程在ModelArts侧完成并上传OBS
3. ais-bench-stubs从OBS上获取统计数据，并上报给Tester

### 训练代码来源

本例中训练代码来自于mindspore的model_zoo：

https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/official/cv/resnet

同时进行了一定的适配修改，主要适配点是适配ModelArts、数据统计上传

### 环境依赖
1. 本程序需要安装 easydict程序包

pip3.7 install easydict

2. 安装modelarts sdk程序包,参考如下网页
https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0004.html#modelarts_04_0004__section16657165520146

### 配置文件详解

配置文件用于配置该次训练任务所需的信息，路径位于Ais-Benchmark-Stubs-x86_64/code/config/modelarts_config.py，填写指导如下：
请参考配置文件说明填写

超参配置参考
resnet 1.3
    'hyperparameters': [
        {'label': 'config_path', 'value': 'resnet50_imagenet2012_Acc_config.yaml'},
        {'label': 'enable_modelarts', 'value': 'True'},
        {'label': 'run_distribute', 'value': 'True'},
        {'label': 'epoch_size', 'value': '5'},
        {'label': 'device_num', 'value': '8'},
        {'label': 'run_eval', 'value': 'True'},
    ],

resnet 1.5
    'hyperparameters': [
        {'label': 'config_path', 'value': 'resnet50_imagenet2012_Boost_config.yaml'},
        {'label': 'enable_modelarts', 'value': 'True'},
        {'label': 'run_distribute', 'value': 'True'},
        {'label': 'epoch_size', 'value': '5'},
        {'label': 'device_num', 'value': '8'},
        {'label': 'run_eval', 'value': 'True'},
    ],

### 单服务器模式  
    单服务器模式指运行n个设备。但是运行是各自设备进行单设备8卡进行业务训练，如果需要打开该模式，需要在config.sh中 增加如下宏设置  
export SINGLESERVER_MODE=True  