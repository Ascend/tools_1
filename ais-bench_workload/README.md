# ais-bench-workload

## 介绍
ais-bench-workload

## 构建教程
### 1. 下载tools仓库代码
+ 通过git clone 命令方式：
```
    git clone https://gitee.com/ascend/tools.git
```

+ 在线下载
打开仓库网页链接，https://gitee.com/ascend/tools， 点击”克隆/下载“，在
弹出的窗口中点击”下载ZIP“，进行下载

### 2. Windows快速构建
快速构建，依赖winrar、git，请确认并安装 
如果下载的是ZIP压缩包，请通过winrar等解压缩工具，解压ZIP包。注意ZIP解压后目录名字是tools-master  
 
在ais-bench_workload/build目录，点击鼠标上下文菜单"git bash here", 进入构建工作目录。鼠标右键没有git相关菜单命令时，请在windows右下角的搜索窗口输入"git" ，找到git  bash，并点击进入。  
构建指令格式：bash ./download_and_build.sh {version} {type}  
参数说明：
+ version 框架版本号。训练模式专用参数。版本取值，对应模型子目录中的适配版本信息
+ type 线上还是线下环境。训练模式专用参数。默认不取值为线下环境。取值为"modelarts"时，表示云上执行训练

构建指令示例：
```
bash ./download_and_build.sh r1.7
bash ./download_and_build.sh r1.7 modelarts
```
构建输出目录ais-bench_workload/output，有4个测试包生成

### 3. 标准构建
#### 3.1. 下载ais-bench stubs测试工具
点击[面向人工智能基础技术及应用的检验检测基础服务平台](http://www.aipubservice.com/#/show/compliance/detail/127)网址, 通过“成果展示”->“标准符合性测试”->“人工智能服务器系统性能测试”， 进入“人工智能服务器系统性能测试”页面，在“测试工具”章节下载Stubs压缩包到本地备用。

#### 3.2. 解压stubs压缩包，将stubs二进制压缩包拷贝到build目录
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

#### 3.3. 构建测试包
工作目录：ais-bench_workload/build

#### 3.4 构建指令
格式：./build.sh  {$stubs_file} {mode} {secondary-folder-name} {third-folder-name} {version} {type}
输出：在ais-bench_workload\output目录会生成相应程序包。
参数说明：
+ stubs_file 下载的stubs.rar中适用构建平台要求的stubs二进制压缩包。
+ mode  执行模式。取值：train--训练模式，inference--推理模式。ais-bench_workload\src目录下一级子目录名称，不包含common。
+ secondary-folder-name 二级子目录名称。ais-bench_workload\src目录下二级子目录名称
+ third-folder-name 三级子目录名称。ais-bench_workload\src目录下三级子目录名称
+ version  框架版本号。训练模式专用参数。版本取值，对应模型子目录中的适配版本信息
+ type 线上还是线下环境。训练模式专用参数。默认不取值为线下环境。取值为"modelarts"时，表示云上执行训练
##### 3.4.1 训练构建指令示例
+ 构建mindspore框架resnet模型 r1.7版本 aarch64架构的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_resnet r1.7
+ 构建mindspore框架resnet模型 r1.7版本 aarch64架构 modelarts运行的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_resnet r1.7 modelarts
+ 构建mindspore框架resnet模型 r1.7版本 x86_64架构的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_resnet r1.7
+ 构建mindspore框架resnet模型 r1.7版本  x86_64架构 modelarts运行的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_resnet r1.7 modelarts
+ 构建mindspore框架bert模型 r1.7版本 aarch64架构的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_bert r1.7
+ 构建mindspore框架bert模型 r1.7版本 aarch64架构 modelarts运行的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-aarch64-1.0.tar.gz train huawei train_mindspore_bert r1.7 modelarts
+ 构建mindspore框架bert模型 r1.7版本 x86_64架构的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_bert r1.7
+ 构建mindspore框架bert模型 r1.7版本  x86_64架构 modelarts运行的程序包
  ./build.sh  ./Ais-Benchmark-Stubs-x86_64-1.0.tar.gz train huawei train_mindspore_bert r1.7 modelarts

##### 3.4.2 推理构建指令示例
+ 构建vision分类classification_and_detection类型，aarch64架构的推理程序包
  ./build.sh ../output/Ais-Benchmark-Stubs-aarch64-1.0.tar.gz inference vision classification_and_detection
+ 构建vision分类classification_and_detection类型，x86_64架构的推理程序包
  ./build.sh ../output/Ais-Benchmark-Stubs-x86_64-1.0.tar.gz inference vision classification_and_detection
+ 构建language分类bert模型, aarch64架构的推理程序包
  ./build.sh ../output/Ais-Benchmark-Stubs-aarch64-1.0.tar.gz inference language bert
+ 构建language分类bert模型, x86_64架构的推理程序包
  ./build.sh ../output/Ais-Benchmark-Stubs-x86_64-1.0.tar.gz inference language bert

## 执行
### 解压测试包
tar -xzvf XXX.tar.gz
说明： XXX.tar.gz是构建教程步骤4构建的测试包
### 执行配置
训练和推理执行之前，请根据相应的指导文档"code/README.md"进行相关配置。
对于训练，还有"code/doc"目录的指导文档可以参考。
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

