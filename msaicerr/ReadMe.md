# Aicore Error分析工具
## 概述
在执行训练发现AI Core error问题时，使用AI Core Error Analyzer工具可以自动快速准确地收集定位AI Core Error问题所需的关键信息，提升开发者对AI Core Error的排查效率。

## 约束
- 该工具当前不支持在容器中部署使用。
- 命令行部署该工具仅支持本地分析使用，即部署该工具的环境应该和日志所在环境为同一环境。
- 暂不支持推理场景。

## 前置条件
1. 进行模型训练时，发现了AI Core Error问题。
2. 使用[npucollector](https://gitee.com/ascend/tools/tree/master/npucollector)工具，收集aicore error时的信息。
3. 使用AI Core Error分析工具解析问题。

## 工具使用方法
1. 获取npucollector收集的tar包，例如gatherv2_aicerr.tar.gz
2. 调用本工具解析，解析指令：
```python3 msaicerr.py -f gatherv2_aicerr.tar.gz```
3. 针对aicore error，会生成相应的info.txt，开发者可根据info.txt进行异常分析。

## 问题分析和定位
本程序运行完毕，会打印Write summary xxxx/info.txt, 用户可以直接通过info.txt文件进行问题分析和定位。
关键信息说明：

***********************  4. Input and output of node *************************

本环节用于分析aicore error时用的地址是否越界。

- 获取aicore error时input output的地址信息
- 获取aicore error时系统分配的内存范围。
- 检查input、output地址信息是否在地址范围内。若不在，则报错。

*********************** 7. result of single_op_test *************************

为保证本环节顺利运行，需保证环境中有真实的device在线，并安装opp、runtime、compiler包为程序运行提供支持。

本环节用于获取单算子运行参数并运行。运行流程如下：
- 通过日志获取异常算子的输入shape、format等编译信息。
- 通过dump数据获取异常算子的输入数据。
- 获取kernel_meta目录下异常算子的.o .json信息。
- 调用rts接口调用单算子运行。

若运行成功，说明整网中发生的aicore error无法单算子复现，此时重点查看4.中地址是否存在异常。

若运行失败，抛出错误码为runtime 0x7开头错误码，则说明aicore error问题复现。此时可使用生成的单算子脚本进行分析。

若抛出其他异常，可具体分析。
