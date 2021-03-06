# Ais-Bench+Mindspore+Bert使用说明

## 1.简介

AI Server Benchmark 是按《信息技术 人工智能 服务器系统性能测试规范》对人工智能服务器系统的性能进行性能评估的测试系统（测试套件），简称Ais-Bench软件。

## 2.使用前提

本程序包运行需要基于以下前提

1. Atlas 800-9000设备
2. 安装好CANN包和Mindspore对应版本。并可以运行正常mindspore测试程序。
3. 保存数据集和相关预处理文件等到设备中。

## 3.集群节点配置

### 3.1 rank_table文件

单机或集群rank_table文件生成方法，请参照[这里](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools#merge_hccl)。

示例：rank_table_16p_64_66.json

```bash
   "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "90.90.66.64",
            "device": [
                {"device_id": "0", "device_ip": "192.100.61.14", "rank_id": "0"},
                {"device_id": "1", "device_ip": "192.100.62.14", "rank_id": "1"},
                {"device_id": "2", "device_ip": "192.100.63.14", "rank_id": "2"},
                {"device_id": "3", "device_ip": "192.100.64.14", "rank_id": "3"},
                {"device_id": "4", "device_ip": "192.100.61.15", "rank_id": "4"},
                {"device_id": "5", "device_ip": "192.100.62.15", "rank_id": "5"},
                {"device_id": "6", "device_ip": "192.100.63.15", "rank_id": "6"},
                {"device_id": "7", "device_ip": "192.100.64.15", "rank_id": "7"}
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "90.90.66.66",
            "device": [
                {"device_id": "0", "device_ip": "192.100.61.16", "rank_id": "8"},
                {"device_id": "1", "device_ip": "192.100.62.16", "rank_id": "9"},
                {"device_id": "2", "device_ip": "192.100.63.16", "rank_id": "10"},
                {"device_id": "3", "device_ip": "192.100.64.16", "rank_id": "11"},
                {"device_id": "4", "device_ip": "192.100.61.17", "rank_id": "12"},
                {"device_id": "5", "device_ip": "192.100.62.17", "rank_id": "13"},
                {"device_id": "6", "device_ip": "192.100.63.17", "rank_id": "14"},
                {"device_id": "7", "device_ip": "192.100.64.17", "rank_id": "15"}
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 3.2 ssh节点文件

如果运行设备大于1个设备，那么需要运行设置ssh节点文件。说明节点信息

```bash
{
    "cluster": {
       "90.90.66.62": {    # 节点ip 必须与ranktable中对应
       "user": "root",   # 用户名 免密可以不用设置
       "pd": "xx",  # 密码 免密不用设置
       "port":0   # 端口 默认22  可以不用设置
     },
     "90.90.66.64": {
        "user": "root",
        "pd": "xx",
        "port":1
     }
   }
}
```

## 4.集群节点免密设置

设置密钥认证的参考操作如下：

+ ssh-keygen -t rsa -b 2048   # 登录管理节点并生成SSH Key。安全起见，建议用户到"Enter passphrase"步骤时输入密钥密码，且符合密码复杂度要求。建议执行这条命令前先将umask设置为0077，执行完后再恢复原来umask值。
+ ssh-copy-id -i ~/.ssh/id_rsa.pub `<user>`@`<ip>`   # 将管理节点的公钥拷贝到所有节点的机器上，`<user>`@`<ip>`替换成要拷贝到的对应节点的账户和ip。
+ 设置ssh代理管理ssh密钥，避免工具批量安装操作过程中输入密钥密码和节点密码

```
    ssh-agent bash   # 开启ssh-agent的bash进程
    sh-add                # 向ssh-agent添加私钥
```

## 5.config.sh配置文件信息

> #python版本设置
> export PYTHON_COMMAND=python3.7.5
> #训练数据集路径
> export TRAIN_DATA_PATH=/home/datasets/Bert-Dataset/
> #验证数据集路径
> export EVAL_DATA_PATH=/home/datasets/Bert-TestData/
> #预训练模型
> export PRETRAIN_MODEL_PATH=/home/models/ms_bert_large.ckpt

> export EPOCH_SIZE=5
> export TRAIN_STEPS=12000

> 节点信息
> export RANK_SIZE=8
> export DEVICE_NUM=8

> #options needed only if rank_size > 1
> export RANK_TABLE_FILE=/home/tools/rank_table_8p.json

> #needed only in cluster mode
> export NODEINFO_FILE=/home/tools/ssh64_66.json

说明：
配置文件默认是8卡训练。
单卡训练时，需要设置RANK_SIZE=1，DEVICE_NUM=1，且不能使用RANK_TABLE_FILE环境变量.
同时还请按需增加指定执行卡序号变量声明export SINGLE_CARD_INDEX。默认 SINGLE_CARD_INDEX=0，可以不显式声明。其它卡时需要显式声明，比如export SINGLE_CARD_INDEX=6
