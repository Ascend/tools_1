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
[WARNING] 1. visit https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench to install
[WARNING] 2. or run cmd: pip3  install -v --force-reinstall 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend' to install
```
**故障原因：**  
环境安装低版本aclruntime, 推理工具运行时使用的是高版本的ais_bench

**处理步骤：**  
更新aclruntime程序包  
## 3. 推理工具组合输入进行推理时遇到"save out files error"
**故障现象**
```bash
[ERROR] save out files error array shape:(1, 349184, 2) filesinfo:[['prep/2002_07_19_big_img_18.bin', 'prep/2002_07_19_big_img_90.bin', 'prep/  2002_07_19_big_img_130.bin', 'prep/2002_07_19_big_img_135.bin', 'prep/  2002_07_19_big_img_141.bin', 'prep/2002_07_19_big_img_158.bin', 'prep/  2002_07_19_big_img_160.bin', 'prep/2002_07_19_big_img_198.bin', 'prep/  2002_07_19_big_img_209.bin', 'prep/2002_07_19_big_img_230.bin', 'prep/  2002_07_19_big_img_247.bin', 'prep/2002_07_19_big_img_254.bin', 'prep/  2002_07_19_big_img_255.bin', 'prep/2002_07_19_big_img_269.bin', 'prep/  2002_07_19_big_img_278.bin', 'prep/2002_07_19_big_img_300.bin']]  files_count_perbatch:16 ndata.shape0:1
```
**故障原因**
input文件由16个文件组成，推理输出进行结果文件切分时，默认按shape的第一维切分，而shape最高维度是1，不是16的倍数。所以报错
**处理步骤**
推理工具参数"--output_batchsize_axis"取值为1。 改成以shape第2维进行切分
