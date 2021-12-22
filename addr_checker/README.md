# 地址异常检测工具
## 工具介绍
为提高系统故障维测效率，提供地址异常检测工具。算子地址异常是导致aicore error的常见原因。使用本工具可以快速获取系统异常时，系统分配的内存区间以及算子input、output所申请的地址，并进行地址比对。如果出现地址异常，可提示开发者进行内存异常定位。
## 使用方法
运行用户脚本，发生aicore error时，使用本工具。输入参数 *-p*或者 *--plog_path*, 后跟算子host日志路径。
```
python3.7 addr_checker.py -h
usage: addr_checker.py [-h] -p PLOG_PATH

optional arguments:
  -h, --help            show this help message and exit
  -p PLOG_PATH, --plog_path PLOG_PATH
                        <Required> the plog path

```
示例:
```
python3.7 addr_checker.py -p /root/ascend/log/plog
```
*注意:使用本工具解析时，注意不要有上次的残留日志。建议采用本仓中[npucollector](https://gitee.com/liuzhenyuhw/tools/tree/master/npucollector)进行采集。*
## 结果解析
结果一：算子存在地址异常。则需要继续定位算子地址异常原因。
示例：
```
```
结果二：算子地址不存在异常。则排除非算子input output异常，需要进一步定位原因。

示例：
```
```