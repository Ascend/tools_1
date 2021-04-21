#!/bin/bash
modules=(core ge log ops install)

#clear previous process
if ps aux | grep .sh | grep running_process;then
    ps aux | grep .sh | grep running_process | awk '{print $2}' | xargs kill -9
fi

#argv check
if [ $# -ne 2 ];then
    echo "[error] argument must be 2"
    echo "argument1: [cmd] cmd which execute origin task, including origin args, should be embraced by \""
    echo "argument2: [tar_file_name] file name with absolute path which info will compress into. suffix must use .tar.gz"
    echo "example: bash npucollect.sh \"echo 123\" /home/target.tar.gz"
    exit
fi

cmd=$1
path=${2%.tar.gz*}

if ! grep -q -E '\.tar\.gz$' <<< "$2";then
    echo "[error] argument2 invalid, suffix must use .tar.gz"
    echo "argument1: [cmd] cmd which execute origin task, including origin args, should be embraced by \""
    echo "argument2: [tar_file_name] file name with absolute path which info will compress into. suffix must use .tar.gz"
    echo "example: bash npucollect.sh \"echo 123\" /home/target.tar.gz"
    exit
fi

if ! grep -q -E '^/' <<< "$2";then
    echo "[error] argument2 invalid, prefix must use /, must use absolute path"
    echo "argument1: [cmd] cmd which execute origin task, including origin args, should be embraced by \""
    echo "argument2: [tar_file_name] file name with absolute path which info will compress into. suffix must use .tar.gz"
    echo "example: bash npucollect.sh \"echo 123\" /home/target.tar.gz"
    exit
fi

#path check and clear
if [ -d $path ];then
    touch $path/check
    if [ $? -ne 0 ];then
        echo "[error] path:$path permission limit, can't create file"
        echo "argument1: [cmd] cmd which execute origin task, including origin args, should be embraced by \""
        echo "argument2: [tar_file_name] file name with absolute path which info will compress into. suffix must use .tar.gz"
        echo "example: bash npucollect.sh \"echo 123\" /home/target.tar.gz"
        exit
    fi
else
    mkdir -p $path
    if [ $? -ne 0 ];then
        echo "[error] path:$path permission limit, can't create dir"
        echo "argument1: [cmd] cmd which execute origin task, including origin args, should be embraced by \""
        echo "argument2: [tar_file_name] file name with absolute path which info will compress into. suffix must use .tar.gz"
        echo "example: bash npucollect.sh \"echo 123\" /home/target.tar.gz"
        exit
    fi
fi

path_prefix=${0%npucollect.sh*}

trap "
if ps aux | grep .sh | grep running_process;then
    ps aux | grep .sh | grep running_process | awk '{print \$2}' | xargs kill -9
    bash ${path_prefix}core.sh post_process $path
    rm -rf $path
fi
exit
" EXIT


echo "--------pre process start--------"
#pre action
pre_name=pre_process
for module in ${modules[*]}
do
    echo "----------[$module] pre process start------"
    bash $path_prefix$module.sh $pre_name $path
    echo "----------[$module] pre process end------"
done
echo "--------pre process end--------"

echo "--------running process start--------"
#running action
running_pid=()
running_name=running_process
for module in ${modules[*]}
do
    echo "----------[$module] running process start------"
    bash $path_prefix$module.sh $running_name $path &
    running_pid[${#running_pid[@]}]=$!
    echo "----------[$module] running process end------"
done
echo "--------running process end--------"

export PRINT_MODEL=1
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=2
export NPU_COLLECT_PATH=$path
export ASCEND_GLOBAL_LOG_LEVEL=1

ulimit -c unlimited
$cmd > $path/log/host/screen.txt 2>&1

for pid in ${running_pid[*]}
do
    kill -0 $pid
    if [ $? -eq 0 ]; then
        kill -9 $pid>/dev/null 2>&1
    fi
done

running_once_name=running_process_once
for module in ${modules[*]}
do
    echo "----------[$module] running process once start------"
    bash $path_prefix$module.sh $running_once_name $path
    echo "----------[$module] running process once end------"
done

echo "--------post process start--------"
#post action
post_name=post_process
for module in ${modules[*]}
do
    echo "----------[$module] post process start------"
    bash $path_prefix$module.sh $post_name $path
    echo "----------[$module] post process end------"
done
echo "--------post process end--------"

echo "--------compressing--------"
if [ -f $2 ];then
    rm -rf $2
fi
cd ${path%/*}
tar -zcvf ${2##*/} ${path##*/}

if [ $? -eq 0 ];then
    rm -rf $path
else
    echo "[ERROR] tar command not success, origin directory will reserve, please check data in directory or try again"
fi
echo "--------compress done--------"