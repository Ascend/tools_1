## 基于ModelArts的集群训练Ais-Bench接入

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

https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/official/nlp/bert

### 环境依赖
1. 本程序需要安装 easydict程序包

pip3.7 install easydict

2. 安装modelarts sdk程序包,参考如下网页
https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0004.html#modelarts_04_0004__section16657165520146

3. windows10环境运行modelarts时，需要开启WSL2，并安装Ubuntu 20.04.4 LTS。 实现过程，请参照[这里](https://blog.csdn.net/li1325169021/article/details/124285018)

### 配置文件详解

配置文件用于配置该次训练任务所需的信息，路径位于Ais-Benchmark-Stubs-x86_64/code/config/modelarts_config.py，填写指导如下：

超参配置参考
resnet 1.3
    'hyperparameters': [
        {'label': 'enable_modelarts', 'value': 'True'},
        {'label': 'distribute', 'value': 'true'},
        {'label': 'epoch_size', 'value': '2'},      # 训练的epoch数 优先级低于train_steps，如果存在train_steps以此为准，否则以epoch_size为准
        {'label': 'enable_save_ckpt', 'value': 'true'},
        {'label': 'enable_lossscale', 'value': 'true'},
        {'label': 'do_shuffle', 'value': 'true'},
        {'label': 'enable_data_sink', 'value': 'true'},
        {'label': 'data_sink_steps', 'value': '100'},
        {'label': 'accumulation_steps', 'value': '1'},
        {'label': 'save_checkpoint_steps', 'value': '100'}, #表示训练的保存ckpt的step数 建议与train_steps保持一致
        {'label': 'save_checkpoint_num', 'value': '1'},
        {'label': 'train_steps', 'value': '100'},       # 表示训练的step数
        {'label': 'bert_network', 'value': 'large_acc'},
    ],
resnet 1.5
    'hyperparameters': [
        {'label': 'config_path', 'value': 'pretrain_config_Ascend_Boost.yaml'},
        {'label': 'enable_modelarts', 'value': 'True'},
        {'label': 'distribute', 'value': 'true'},
        {'label': 'epoch_size', 'value': '2'},
        {'label': 'enable_save_ckpt', 'value': 'true'},
        {'label': 'enable_lossscale', 'value': 'true'},
        {'label': 'do_shuffle', 'value': 'true'},
        {'label': 'enable_data_sink', 'value': 'true'},
        {'label': 'data_sink_steps', 'value': '100'},
        {'label': 'accumulation_steps', 'value': '1'},
        {'label': 'save_checkpoint_steps', 'value': '99'},
        {'label': 'save_checkpoint_num', 'value': '1'},
        {'label': 'train_steps', 'value': '100'},
    ],

### 单服务器模式  
    单服务器模式指运行n个设备。但是运行是各自设备进行单设备8卡进行业务训练，如果需要打开该模式，需要在config.sh中 增加如下宏设置  
export SINGLESERVER_MODE=True  
