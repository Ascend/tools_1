# ptdbg_ascend

## 环境准备

安装了Pytorch 1.8 或者 Pytorch 1.11版本的Linux系统。

- Linux OS

## 安装

### 从源码安装

您可以从源代码构建 ptdbg_ascend 软件包并将其安装在带NPU或者GPU的AI计算环境上。
> ptdbg_ascend 与 Pytorch 有严格的版本配套关系，从源码构建前，您需要确保已经正确安装了[Pytorch v1.8 或 V1.11 版本](https://www.pytorch.org) 。

#### 下载源码

```
git clone https://gitee.com/ascend/tools.git
cd tools/ptdbg_ascend
```

#### 配置安装环境

```BASH
bash ./configure
```

默认情况下，执行上述命会弹出如下的交互式会话窗口
> 您的会话可能有所不同。

```BASH
Please specify the location of python with available pytorch v1.8/v1.11 site-packages installed. [Default is /usr/bin/python3]
(You can make this quiet by set env [ADAPTER_TARGET_PYTHON_PATH]):
```

此时，要求您输入安装了 Pytorch v1.8或者v1.11 版本的python解释器路径，如果默认路径是正确的，直接回车，否则请输入正确的 python 解释器路径。
> 您可以通过设置 ADAPTER_TARGET_PYTHON_PATH的环境变量，来抑制交互式窗口弹出，但是要确保路径是有效的，否则，仍然会要求您输入正确的 python 解释器路径。

键入后，会耗费几秒钟以确保您的输入是有效的，配置完成后会输出如下提示信息。
```BASH
Configuration finished
```

#### 配置cmake

> 根据您的网络状况，可能需要数分钟来下载ptdbg_ascend的依赖项目以完成配置。

```
mkdir build
cd build
cmake ..
```

#### 执行编译

> 您应当根据实际编译环境，设置合适的并发编译数以提升编译速度。

```BASH
make -j8
```

编译结束后，安装包会生成在

```
./ptdbg_ascend/dist/ptdbg_ascend-0.1-py3-none-any.whl
```

#### 安装

您可以继续执行

```BASH
make install
```

将ptdbg_ascend安装到配置时指定的 python 解释器包目录下，或者使用 pip3 安装 ptdbg_ascend 到您期望的位置。

```
pip3 install ./ptdbg_ascend/dist/ptdbg_ascend-0.1-py3-none-any.whl --upgrade --force-reinstall
```

#### 接口函数

接口函数用于dump过程的配置，如下：

| 函数          | 描述                                                                                                |
|-------------|---------------------------------------------------------------------------------------------------|
| set_dump_path | 用于设置dump文件的路径(包含文件名)，参数示例：“/var/log/dump/npu_dump.pkl”                                            |
| set_dump_switch | 设置dump使能开关，不设置则默认处于关闭状态。参数为：“ON” 或者 "OFF"                                                                      |
| seed_all    | 固定随机数，参数为随机数种子，默认种子为：1234.                                                                        |
| register_hook | 用于注册dump回调函数，例如：注册精度比对hook：register_hook(model, acc_cmp_dump).                               |
| compare     | 比对接口，将GPU/CPU/NPU的dump文件进行比对，第三个参数为存放比对结果的目录；<br/>文件名称基于时间戳自动生成，格式为：compare_result_timestamp.csv. |

#### 使用示例

以训练场景为例，在你需要dump数据的step启动之前，设置使能开关和dump路径。
```
from ptdbg_ascend import *

# 在训练/推理开始前固定随机数
seed_all()
# 设置dump路径（含文件名）
set_dump_path("./npu_dump.pkl")
”“”
# 对模型注入精度比对的hook,第三个参数为dump模式
# dump模式有三种：
    "SUMMERY": 1, 摘要模式，每个tensor dump最多10个数，同时dump出tensor的sum, mean;
    "SAMPLE":  2, 采样模式，tensor数据按16倍下采样后dump
    "ALL":     3  全dump，dump出tensor的完整数据
“”“
register_hook(model, acc_cmp_dump, dump_mode=1)

# 注册溢出检测回调函数，只有两个参数（NPU场景,GPU和CPU不支持）
register_hook(model, overflow_check)

# dump开关默认处于关闭状态.
# 如果需要全量dump，建议在注册hook之前就将开关打开；
# 如果只在特定的step dump，则在期望dump的迭代开始前打开dump开关
set_dump_switch("ON")
...
# 在期望dump的迭代结束后关闭dump开关
set_dump_switch("OFF")

# 数据dump完成后,比对dump的NPU vs GPU/CPU数据, 第三个参数中的目录必须是已存在的目录
compare("./npu_dump.pkl", "./gpu_dump.pkl", "./output", True)
```

## 贡献

psuh代码前，请务必保证已经完成了基础功能测试和网络测试！

## Release Notes

Release Notes请参考[RELEASE](RELEASE.md).
