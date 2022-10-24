
msprof_bin="/home/infname46/Ascend/ascend-toolkit/5.1.RC2/tools/profiler/bin/msprof"

output="/home/yhz/SiamFC/lcmout"
rm -rf $output/*

msprof_py='python3 /home/infname46/Ascend/ascend-toolkit/5.1.RC2/tools/profiler/profiler_tool/analysis/msprof/msprof.py'

model="/home/yhz/SiamFC/om/exemplar_bs1.om"
input="/home/yhz/SiamFC/pre_dataset1_500/"
#app="./python3.7 /home/infname46/lcm/code/tools_master/ais-bench_workload/tool/ais_infer/ais_infer.py --model $model --device 0 --debug=1 --input=$input --output ./lcmout/"
app="/home/lipan/tools/msame/out/msame --model $model --device 0 --debug=1  --output $output --input $input"

$msprof_bin --output=$output --application="/home/lipan/tools/msame/out/msame --model $model --device 0 --debug=1  --output $output --input $input"  \
--sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on \
| tee -a ./lcmout/p.log

cat ./lcmout/p.log  | grep "Inference time:" | awk '{print $3}' | sort | tail -n 10

str=`cat ./lcmout/p.log  | grep NA  | awk '{print $8}'`
array=(${str//,/ })

profiler_path=`find $output/ -name device_0`
for var in ${array[@]}
do
   echo $var
   ${msprof_py} export timeline -dir $profiler_path --iteration-id $var
   ${msprof_py} export summary -dir $profiler_path --iteration-id $var
done

#$msprof_bin --output=$output --application=\"$app\" --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on

# cmd="$msprof_bin --output=$output --application=\"$app\" --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on"

# $cmd