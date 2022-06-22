# ais-bench-workload


## 介绍

ais-bench-workload

## 构建教程
### 1. 配置本地设备构建环境，通过git clone 命令下载tools仓库代码
```
    git clone https://gitee.com/ascend/tools.git
```

### 2. 下载ais-bench stubs测试工具

点击[面向人工智能基础技术及应用的检验检测基础服务平台](http://www.aipubservice.com/#/show/compliance/detail/127)网址, 通过“成果展示”->“标准符合性测试”->“人工智能服务器系统性能测试”， 进入“人工智能服务器系统性能测试”页面，在“测试工具”章节下载Stubs压缩包到本地备用。

### 3. 解压stubs压缩包，将stubs二进制压缩包拷贝到build目录

结果如下：
```
tools
├── ais-bench_workload
    ├── build
        ├── build.sh
        ├── download_and_build.sh
        ├── Ais-Benchmark-Stubs-aarch64-1.0.tar.gz
        └── Ais-Benchmark-Stubs-x86_64-1.0.tar.gz
```

### 4. 构建测试包
工作目录：ais-bench_workload/build

#### 4.1 构建训练测试包
构建指令格式：./build.sh  {$stubs_file} train {company} {model_name} {version} {environment}

参数说明：
+ stubs_file 下载的stubs.rar中适用构建平台要求的stubs二进制压缩包。
+ train  训练模式。

+ company 公司名称。目前支持hauwei、nvidia。
+ model_name 训练模型名称，如 train_mindspore_resnet
+ version  框架版本号。比如取值r1.7
+ environment 线上还是线下环境。默认不取值为线下环境。取值为"modelarts"时，表示云上执行训练


构建指令示例：

+ ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_resnet r1.7
+ ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_resnet r1.7 modelarts
+ ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_resnet r1.7
+ ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_resnet r1.7 modelarts
+ ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_bert r1.7
+ ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_bert r1.7 modelarts
+ ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_bert r1.7
+ ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_bert r1.7 modelarts

output目录生成以下构建测试包：

+ train_huawei_train_mindspore_resnet-Ais-Benchmark-Stubs-aarch64-1.0-r1.7.tar.gz
+ train_huawei_train_mindspore_resnet-Ais-Benchmark-Stubs-aarch64-1.0-r1.7_modelarts.tar.gz
+ train_huawei_train_mindspore_bert-Ais-Benchmark-Stubs-aarch64-1.0-r1.7.tar.gz
+ train_huawei_train_mindspore_bert-Ais-Benchmark-Stubs-aarch64-1.0-r1.7_modelarts.tar.gz
+ train_huawei_train_mindspore_resnet-Ais-Benchmark-Stubs-x86_64-1.0-r1.7.tar.gz
+ train_huawei_train_mindspore_resnet-Ais-Benchmark-Stubs-x86_64-1.0-r1.7_modelarts.tar.gz
+ train_huawei_train_mindspore_bert-Ais-Benchmark-Stubs-x86_64-1.0-r1.7.tar.gz
+ train_huawei_train_mindspore_bert-Ais-Benchmark-Stubs-x86_64-1.0-r1.7_modelarts.tar.gz
#### 4.2 构建推理测试包
构建命令：
```bash
./build.sh $stubs_file inference vision classification_and_detection
./build/build.sh $stubs_file inference language bert
```
说明：
 + 目前支持language类的bert模型、vision类的classification_and_detection推理
+ $stubs_file是指步骤3中Ais-Benchmark-Stubs-aarch64-1.0.tar.gz或Ais-Benchmark-Stubs-x86_64-1.0.tar.gz的路径

output目录生成以下构建测试包：
inference_language_bert-Ais-Benchmark-Stubs-aarch64-1.0.tar.gz
inference_language_bert-Ais-Benchmark-Stubs-x86_64-1.0.tar.gz
inference_vision_classification_and_detection-Ais-Benchmark-Stubs-aarch64-1.0.tar.gz
inference_vision_classification_and_detection-Ais-Benchmark-Stubs-x86_64-1.0.tar.gz

## 执行
### 解压测试包
tar -xzvf XXX.tar.gz
说明： XXX.tar.gz是测试包的压缩包
### 执行配置
训练和推理执行之前，请根据相应的指导文档"code/README.md"进行相关配置。
对于训练，还有"code/doc"目录的指导文档可以参考。
#### 配置code/config.json
注意，对于推理，需要更新code/config.json文件中“Mode”字段为“inference",否则推理结果上报信息中会出现文字不匹配字样，比如“train_result_info”。
#### 设置日志级别

日志级别说明：
+ GLOG日志级别 INFO、 WARNING、 ERROR、FATAL对应的值分别为0、1、2、3.

设置指令： export GLOG_v=3
#####  训练日志

+ 对于modelarts训练，在code/code/ma-pre-start.sh中设置
+ 对于非modelarts训练，在code/common/mindspore_env.sh中设置

##### 推理日志
+ 在code/config/config.sh中设置

### 执行推理或训练
请参照测试包中code/README.md介绍的推理或训练执行方法进行推理训练。
## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](../CONTRIBUTING.md)。

## 许可证
[Apache License 2.0](LICENSE)

