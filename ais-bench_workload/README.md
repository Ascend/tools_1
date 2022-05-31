# ais-bench-workload

## 介绍

ais-bench-workload

## 脚本文件说明


## 构建教程
1.配置本地设备构建环境，通过git clone 命令下载tools仓库代码
```
    git clone https://gitee.com/ascend/tools.git
```
2.下载stub二进制文件

点击[面向人工智能基础技术及应用的检验检测基础服务平台](http://www.aipubservice.com/#/show/compliance/detail/127)网址, 通过“成果展示”->“标准符合性测试”->“人工智能服务器系统性能测试”， 进入“人工智能服务器系统性能测试”页面，在“测试工具”章节下载Stubs压缩包到本地备用。


3.解压stubs压缩包，将stubs二进制压缩包拷贝到build目录

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

4.构建测试包

4.1 构建训练测试包

以下示例以X86_64环境来说明。工作目录：build
构建指令格式：./build.sh  {$stubs_file} {mode} {company} {model_name} {version} {environment}
参数说明：
+ stubs_file 下载的stubs.rar中适用构建平台要求的stubs二进制压缩包。
+ mode  功能分类，目前支持train、inference。

+ company 公司名称。目前支持hauwei、nvidia。
+ model_name 训练模型名称，如 train_mindspore_resnet
+ version  版本号。比如取值r1.7
+ environment 线上还是线下环境。默认不取值为线下环境。取值为"modelarts"时，表示云上执行训练

构建命令示例：

+ ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_resnet r1.7
+ ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_bert r1.7



构建结果：

output目录生成以下文件：

train_huawei_train_mindspore_resnet-Ais-Benchmark-Stubs-x86_64-1.0-r1.7.tar.gz
train_huawei_train_mindspore_bert-Ais-Benchmark-Stubs-x86_64-1.0-r1.7.tar.gz

4.2 构建推理测试包（待补充）
## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](../CONTRIBUTING.md)。

## 许可证
[Apache License 2.0](LICENSE)

