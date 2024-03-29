中文|[英文](README_EN.md)

# tools

#### 介绍

Ascend tools，昇腾工具仓库。

**请根据自己的需要进入对应文件夹获取工具，或者点击下面的说明链接选择需要的工具进行使用。**

#### 使用说明

1.  [msame](https://gitee.com/ascend/tools/tree/master/msame)

    **模型推理工具**：输入.om模型和模型所需要的输入bin文件，输出模型的输出数据文件。

2.  [img2bin](https://gitee.com/ascend/tools/tree/master/img2bin)

    **bin文件生成工具**：生成模型推理所需的输入数据，以.bin格式保存。

3.  [makesd](https://gitee.com/ascend/tools/tree/master/makesd)
    
    **制卡工具**：制卡工具包，提供ubuntu下制卡功能。

4.  [configure_usb_ethernet](https://gitee.com/ascend/tools/tree/master/configure_usb_ethernet)  
     **USB虚拟网卡连接脚本**：配置USB网卡对应的IP地址。
    
5. [pt2pb](https://gitee.com/ascend/tools/tree/master/pt2pb)  

   **pytorch模型转tensorflow pb模型工具**：输入pytorch权重参数模型，转为onnx，再转为pb模型

6. [dnmetis](https://gitee.com/ascend/tools/tree/master/dnmetis)  

   **NPU推理精度和性能测试工具**：使用Python封装ACL的C++接口，输入om模型和原始数据集图片、标签，即可执行模型推理，输出精度数据和性能数据  

7. [msquickcmp](https://gitee.com/ascend/tools/tree/master/msquickcmp)    

   **一键式全流程精度比对工具**：该工具适用于tensorflow和onnx模型，输入原始模型和对应的离线om模型，输出精度比对结果。    

8. [precision_tool](https://gitee.com/ascend/tools/tree/master/precision_tool)    
   **精度问题分析工具**：该工具包提供了精度比对常用的功能，当前该工具主要适配Tensorflow训练场景，同时提供Dump数据/图信息的交互式查询和操作入口。 

9. [cann-benchmark_infer_scripts](https://gitee.com/ascend/tools/tree/master/cann-benchmark_infer_scripts)    
    **cann-benchmark推理软件对应的模型前后处理脚本**： 该工具包含cann-benchmark推理工具模型处理脚本, 包括：结果解析脚本和前后处理脚本等。这些脚本需根据cann-benchmark指导手册说明使用。
10. [tfdbg_ascend](https://gitee.com/ascend/tools/tree/master/tfdbg_ascend)    
    **Tensorflow2.x dump工具**：该工具提供CPU/GPU平台上Tensorflow2.x运行时数据Dump能力。

11. [ais-bench_workload](https://gitee.com/ascend/tools/tree/master/ais-bench_workload)    
    **ais-bench_workload**： 该目录包含基于Ais-Bench软件的训练和推理负载程序，用于测试验证。Ais-Bench是基于AI标准针对AI服务器进行性能测试的工具软件。

12. [intelligent_edge_tools](https://gitee.com/ascend/tools/tree/master/intelligent_edge_tools)  
    **intelligent_edge_tools**： 该目录包含智能边缘工具集。
    
13. [auto-optimizer](https://gitee.com/ascend/tools/tree/master/auto-optimizer)  
    **auto-optimizer**： 提供基于ONNX的改图、自动优化及端到端推理流程。
    
13. [saved_model2om](https://gitee.com/ascend/tools/tree/master/saved_model2om)  
    **TensorFlow1.15 saved_model模型转om模型工具**：输入TensorFlow存储的saved_model模型，转换为pb模型，再转换为om模型

14. [mindxedge_whitebox](https://gitee.com/ascend/tools/tree/master/mindxedge_whitebox)  
    **MindXEdge 白牌化安装工具**：支持Atlas500智能小站进行白牌化的首次安装，安装后设备将变为白牌化的设备
#### 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](./CONTRIBUTING.md)。

#### 许可证
[Apache License 2.0](LICENSE)

