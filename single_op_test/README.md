# 单算子运行工具
# 工具介绍
AI网络运行往往是一个非常耗时且占用空间的过程，本工具旨在提取最小单算子用例，缩短定位时间与路径。
# 前置条件
1. 修改用户脚本，获取算子Aicore error时算子的输入数据以及kernel_meta信息。参考链接：[数据准备](https://support.huaweicloud.com/developmenttg-cann502alpha5training/atlasaicerrtrain_16_0004.html)
2. 解析dump数据，将数据解析为npy格式的文件.[参考链接](https://support.huaweicloud.com/auxiliarydevtool-cann502alpha2infer/atlasaccuracy_16_0017.html)
```
python3.7.5 msaccucmp.py convert -d dump_file 
```
PS:推理获取dump数据方式可以参考[链接](https://support.huaweicloud.com/auxiliarydevtool-cann502alpha2infer/atlasaccuracy_16_0005.html)
# 构造用例
方法一：手动构造
1. 根据aicore error的打屏日志或host日志，获取报错算子的类型。通过GE图，获取算子输入shape、dtype、format等信息。构造用例如下:
```
from op_test_frame.ut import OpUT


def run_sample_case():
    ut_case = ReduceOpUT("ReduceSumD", None, None)

    case1 = {
        "params": [
            {"shape": (1, 11, 1, 15, 32), "dtype": "float16", "format": "ND", "ori_shape": (1, 11, 1, 15, 32),
             "ori_format": "ND", "param_type": "input",
             "value": "/home/workspace/PycharmProjects/op-test/input/data/ReduceSumD/Ascend910A/ReduceSumD_static_shape_test_ReduceSumD_auto_case_name_1_input0.bin"},
            {"shape": (1, 15, 32), "dtype": "float16", "format": "ND", "ori_shape": (1, 15, 32),
             "ori_format": "ND", "param_type": "output"},
            (0, 1)
        ],
        "case_name": "test_reduce_mean_1",
        "bin_path": "/home/workspace/PycharmProjects/op-test/input/kernel_meta/ReduceSumD_static_shape_test_ReduceSumD_auto_case_name_1_ascend910a.o"
    }
    ut_case.add_direct_case(case1)
    ut_case.run("all")


if __name__ == '__main__':
    run_sample_case()
```
参数说明：
- "params": 算子输入信息，从GE图中获取。
- "value": 算子输入数据，若不配置，则使用random函数获取range为(0, 1)
- "case_name": 用例名称，开发者自行设置
- "bin_path": 算子.o的名称，需要注意的是，工具使用bin_path替换.o为.json的方式获取json文件

```ut_case.add_direct_case(case1)```方法，用例将会上板运行。

方法二：自动构造(实验性特性)
使用方法如下，示例：

```
python3 op_test.py --graph ge_onnx_build.pbtxt --node  TransData --bin_path /home/workspace/PycharmProjects/op-test/input/kernel_meta/ReduceSumD_static_shape_test_ReduceSumD_auto_case_name_1_ascend910a.o --value_path /home/HwHiAiUser/dumptonumpy/Pooling.pool1.1147.1589195081588018
```
使用说明:
```
python3 op_test.py -h
usage: op_test.py [-h] [--graph GRAPH_NAME] [--node NODE_NAME]
                  [--bin_path BIN_PATH] [--value_path VALUE_PATH]

test single op.

optional arguments:
  -h, --help            show this help message and exit
  --graph GRAPH_NAME    the graph pre_run after build,exp
                        "ge_onnx_00000141_graph_1_PreRunAfterBuild.pbtxt"
  --node NODE_NAME      the node to test, exp "ReduceSum_in_0"
  --bin_path BIN_PATH   the bin to test, exp "kernel_meta/te_transdata_ff9a78f
                        daf5103c60_feb5e077cf4dd9cc.o"
  --value_path VALUE_PATH
                        the dump data, exp "/home/HwHiAiUser/dumptonumpy/Pooli
                        ng.pool1.1147.1589195081588018",value_path +
                        "input.x.npy" wil be used for test

```
参数说明：
- graph GE图，一般选取最后一张pbtxt图
- node  要测试的节点名称，一般从图中node的name属性获取
- bin_path (可选) 配置网络编译时产生的算子.o的路径，注意：同路径下需要有同名的.json文件。若不配置该选项，则会重新编译新的.o .json运行
- value_path(可选) 配置算子的输入数据。若解析出的的数据文件名为/home/HwHiAiUser/dumptonumpy/Pooling.pool1.1147.1589195081588018.input.0.npy，则配置value_path为/home/HwHiAiUser/dumptonumpy/Pooling.pool1.1147.1589195081588018即可，工具会根据input的前后顺序分别获取 /home/HwHiAiUser/dumptonumpy/Pooling.pool1.1147.1589195081588018.input.xxx.npy作为用例输入。
