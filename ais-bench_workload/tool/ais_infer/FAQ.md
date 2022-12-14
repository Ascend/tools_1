# FAQ

## 1. input文件夹输入有很多的数据，如果选择其中某一部分做输入进行推理。比如 input文件夹中有50000张图片，如果只选择其中100张进行推理
----------------------------------------
当前推理工具针对input文件夹中数据是全部读取的，没有读取某部分数据功能

如果需要该功能，可以通过如下脚本命令执行，生成某一部分的软链接的文件夹，传入到推理程序中。

```bash
# 首先搜索src目录下的所有的JPEG的文件  然后选取前100个 然后通过软链接的方式链接dst文件夹中
find ./src -type f -name "*.JPEG" | head -n 100 | xargs -i ln -sf {} ./dst
```

## 2. 推理工具运行时，会出现aclruntime版本不匹配告警
**故障现象**
运行推理工具进行推理时屏幕输出如下告警：
```bash
root#  python3 -m ais_bench --model /home/lhb/code/testdata/resnet50/model/pth_resnet50_bs1.om --loop 2
[WARNING] aclruntime version:0.0.1 is lower please update aclruntime follow any one method
[WARNING] 1. visit https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer to install
[WARNING] 2. or run cmd: pip3  install -v --force-reinstall 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_infer/backend' to install
```
**故障原因：**  
环境安装低版本aclruntime, 推理工具运行时使用的是高版本的ais_bench

**处理步骤：**  
更新aclruntime程序包  