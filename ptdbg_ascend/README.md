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

| 函数              | 描述                                                                                                |
|-----------------|---------------------------------------------------------------------------------------------------|
| set_dump_path   | 用于设置dump文件的路径(包含文件名)，参数示例：“/var/log/dump/npu_dump.pkl”                                            |
| set_dump_switch | 设置dump范围，不设置则默认处于关闭状态。第一个参数为：“ON” 或者 "OFF",若需要控制dump的算子范围，则需要第二、三个参数，默认不配置                        |
| seed_all        | 固定随机数，参数为随机数种子，默认种子为：1234.                                                                        |
| register_hook   | 用于注册dump回调函数，例如：注册精度比对hook：register_hook(model, acc_cmp_dump).                                    |
| compare         | 比对接口，将GPU/CPU/NPU的dump文件进行比对，第三个参数为存放比对结果的目录；<br/>文件名称基于时间戳自动生成，格式为：compare_result_timestamp.csv. |
| parse           | (若pkl文件中有)打印特定api接口的堆栈信息、统计数据信息，第一个参数为pkl文件名，第二个参数为要抽取的api接口前缀，例如"21_Torch_norm".                 |

#### 使用说明
1) seed_all和set_dump_path在训练主函数main一开始就调用，避免随机数固定不全；
2) 模型较大时dump较慢，可以通过两步法：先粗->后精，逐步缩小范围来定位，如下：<br/>
   第一步，通过“下采样dump”做整网比对，快速找到问题的可能点；<br/>
        "下采样dump"会dump采样后的数据，以及完整数据的统计信息：dtype, shape, max, min, mean等；<br/>
   第二步，基于整网比对找到的精度问题产生起始点/起始点范围，再通过指定api dump，或者指定范围做完整数据dump，从而进行精确分析；
3) 指定范围dump的控制方法：
```
# 初步确定了问题的起始范围后，可以通过如下方式dump单API或者小范围API的全量数据
# 实现方式：通过set_dump_switch的第二、第三个参数控制dump的范围

# 示例1： dump指定api/api列表.
set_dump_switch("ON", mode=2, scope=["1478_Tensor_permute", "1484_Tensor_transpose", "1792_Torch_relue"])

# 示例2： dump指定范围. 会dump 1000_Tensor_abs 到 1484_Tensor_transpose_forward之间的所有api
set_dump_switch("ON", mode=3, scope=["1000_Tensor_abs", "1484_Tensor_transpose_forward"])

# 示例3： STACK模式，只dump堆栈信息， 示例中dump "1000_Tensor_abs" 到 "1484_Tensor_transpose_forward" 之间所有api的STACK信息
set_dump_switch("ON", mode=4, scope=["1000_Tensor_abs", "1484_Tensor_transpose_forward"])
```
4) dump数据存盘说明：<br/>

精度比对dump场景 <br/>
假设配置的dump文件名为npu_dump.pkl，此时dump的结果为两部分：
* 文件npu_dump.pkl 中包含dump数据的api名称、dtype、 shape、统计信息：max, min, mean.<br/>
* 文件夹npu_dump_timestamp，文件夹下为numpy格式的dump数据.<br/>

整网dump和指定范围dump结果的区别：
* 指定范围dump时，npu_dump.pkl 中还包含stack信息<br/>

溢出检测dump场景<br/>
测试不需要配置dump文件名，会在当前目录自动生成：
* 溢出检测的pkl文件名格式为Overflow_info_{timestamp}.pkl，每次溢出时时间戳不同<br/>
  pkl文件中包含dump数据的api名称、dtype、 shape(不包含统计信息max, min, mean)。
* 对应的dump数据存放目录为Overflow_info_{timestamp}，dump数据为完整Tensor数据，存放格式为numpy。

#### 场景化示例
#### 场景1：训练场景的精度问题分析
第一步，采样模式下的整网比对，初步定位异常范围<br/>
数据dump。下采样方式dump NPU和GPU/CPU数据，比对双方采样步长要相同，下面以NPU为例（GPU/CPU dump基本相同）：<br/>
```
from ptdbg_ascend import *

# 在main函数开始前固定随机数
seed_all()
# 设置dump路径（含文件名）
set_dump_path("./npu_dump.pkl")

...

# 注册精度比对dump的hook函数
# 第一个参数是model对象， 第二个参数为精度比对dump的钩子函数，必须配置为：acc_cmp_dump，该函数从ptdbg_ascend中import
# 第三个参数为dump的采样步长
# dump数据的采样规则：
     每个tensor dump范围为:(256(前) + 中间部分按采样步长取数 + 256(尾))，同时dump出tensor的max, min, mean;
     采样步长配置：1~100的整数，表示采样的步长; 默认配置为：1，全量dump（无采样）

# 示例,采样步长为2，即间隔一个点抽数，分采样50%
register_hook(model, acc_cmp_dump, dump_step=2)

...

# dump默认处于关闭状态，设置dump开关为打开
# 如果只在特定的step dump，则在期望dump的迭代开始前打开dump开关，step结束后关掉。
set_dump_switch("ON")

...

# 在期望dump的step结束后关闭dump开关
set_dump_switch("OFF")

...

```

比对dump数据<br/>
```
from ptdbg_ascend import *

...

# 数据dump完成后,比对dump的NPU vs GPU/CPU数据, compare第二个参数中的目录必须是已存在的目录
比对示例：
dump_result_param={
"npu_pkl_path": "./npu_dump.pkl",
"bench_pkl_path": "./gpu_dump.pkl",
"npu_dump_data_dir": "./npu_dump_20230104_13434",
"bench_dump_data_dir": "./gpu_dump_20230104_132544"
}
compare(dump_result_param, "./output", True)
```
第二步：缩小范围分析<br/>
      指定api范围做完整数据的dump，此时也可以做精度比对。<br/>
      指定范围dump时，还会dump出stack信息，便于找到api调用点。<br/>
      示例代码中只包含第一步基础之上，需要调整的设置。
```
# 设置dump路径（含文件名），dump路径若不重新设置，会导致整网dump的数据被覆盖
set_dump_path("./npu_dump_scope.pkl")

...

# 注册精度比对dump的hook函数，调整dump_step为1，此时为全量dump
register_hook(model, acc_cmp_dump, dump_step=1)

...

# 通过set_dump_switch控制dump的范围
# 示例1： dump指定api/api列表.
set_dump_switch("ON", mode=2, scope=["1478_Tensor_permute", "1484_Tensor_transpose", "1792_Torch_relue"])
# 示例2： dump指定范围. 会dump 1000_Tensor_abs 到 1484_Tensor_transpose_forward之间的所有api
set_dump_switch("ON", mode=3, scope=["1000_Tensor_abs", "1484_Tensor_transpose_forward"])
...
```
按范围dump后的分析<br/>
可以基于dump的完整数据做比对，可以结合堆栈信息分析代码，也可以做单API模型的问题复现；

#### 场景2：提取指定API的堆栈信息/dump数据的统计信息
指定范围dump的信息可能包含多个api，且pkl文件显示不直观，这里通过parse接口可以清晰的显示特定api的堆栈信息和dump数据统计信息
```
from ptdbg_ascend import *

# 提取dump信息中第21次调用的API：Torch_batch_normal的堆栈信息及数据统计信息
parse("./npu_dump.pkl", "21_Torch_batch_normal")
```

#### 场景3：溢出检测分析（NPU场景,GPU和CPU不支持）
```
from ptdbg_ascend import *

# 在main函数起始位置固定随机数
seed_all()

...

#注册溢出检测的hook：
# 第一个参数是model对象， 第二个参数为精度比对dump的钩子函数名，必须配置为：overflow_check，该函数从ptdbg_ascend中import
# 第三个参数为溢出检测的次数，例如配置为3，表示检测到第三次溢出时停止训练;

# 示例，检测到2次溢出后退出
register_hook(model, overflow_check, overflow_nums=2)

...
```

## 贡献

psuh代码前，请务必保证已经完成了基础功能测试和网络测试！

## Release Notes

Release Notes请参考[RELEASE](RELEASE.md).
