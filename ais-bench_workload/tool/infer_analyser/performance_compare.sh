#!/bin/bash
CUR_PATH=$(dirname $(readlink -f "$0"))

# 返回码
declare -i ret_ok=0
declare -i ret_run_failed=1

check_python_package_is_install()
{
    local PYTHON_COMMAND=$1
    ${PYTHON_COMMAND} -c "import $2" >> /dev/null 2>&1
    ret=$?
    if [ $ret != 0 ]; then
        echo "python package:$1 not install"
        return 1
    fi
    return 0
}

check_args_valid()
{
    [ -f "$MODEL" ] || { echo "model:$MODEL not valid"; return 1; }
    return 0
}

check_env_valid()
{
    check_python_package_is_install ${PYTHON_COMMAND} "aclruntime" \
    || { echo "aclruntime package not install"; return $ret_run_failed;}
}

run_msame()
{
    msame_cmd="$CUR_PATH/../../../msame/out/msame --model $MODEL --output $CACHE_PATH/ --device $DEVICE --loop $LOOP"
    [ "$INPUT" != "" ] && { msame_cmd="$msame_cmd --input $INPUT"; }

    $msame_cmd | tee -a $CACHE_PATH/m.log || { echo "msame run failed"; return $ret_run_failed; }

    #cat $CACHE_PATH/m.log | grep 'Inference time:' | awk '{print $3}' | sed '1d' > $CACHE_PATH/m.times
    cat $CACHE_PATH/m.log | grep 'Inference time:' | awk '{print $3}' > $CACHE_PATH/m.times

    cat $CACHE_PATH/m.log | grep pid | awk '{print $3}' > $CACHE_PATH/m.pid
    $PYTHON_COMMAND $CUR_PATH/info_convert_json.py $CACHE_PATH/m.times $CACHE_PATH/m.pid $CACHE_PATH/msumary.json
}

run_ais_infer()
{
    ais_infer_cmd="$PYTHON_COMMAND $CUR_PATH/../ais_infer/ais_infer.py --model $MODEL --output $CACHE_PATH/ \
        --device $DEVICE --loop $LOOP --warmup_count 0 --output_dirname=aisout"
    [ "$INPUT" != "" ] && { ais_infer_cmd="$ais_infer_cmd --input $INPUT"; }
    $ais_infer_cmd | tee -a $CACHE_PATH/m.log || { echo "ais_infer run failed"; return $ret_run_failed; }
    cp $CACHE_PATH/aisout/sumary.json $CACHE_PATH/asumary.json
}

main()
{
    while [ -n "$1" ]
do
  case "$1" in
    -m|--model)
        MODEL=$2
        shift
        ;;
    -p|--python_command)
        PYTHON_COMMAND=$2
        shift
        ;;
    -input|--input)
        INPUT=$2
        shift
        ;;
    -d|--device)
        DEVICE=${2}
        shift
        ;;
    -loop|--loop)
        LOOP=$2
        shift
        ;;
    -h|--help)
        echo_help;
        exit
        ;;
    *)
        echo "$1 is not an option, please use --help"
        exit 1
        ;;
  esac
  shift
done

    [ "$PYTHON_COMMAND" != "" ] || { PYTHON_COMMAND="python3.7";echo "set default pythoncmd:$PYTHON_COMMAND"; }
    [ "$LOOP" != "" ] || { LOOP="1";echo "set default LOOP:$LOOP"; }
    [ "$DEVICE" != "" ] || { DEVICE="0";echo "set default DEVICE:$DEVICE"; }

    CACHE_PATH=$CUR_PATH/pcache
    [ ! -d $CACHE_PATH ] || rm -rf $CACHE_PATH
    mkdir -p $CACHE_PATH

    check_args_valid || { echo "check args not valid return"; return $ret_run_failed; }
    check_env_valid || { echo "check env not valid return"; return $ret_run_failed; }

    run_ais_infer || { echo "run ais_infer failed"; return $ret_run_failed; }
    run_msame || { echo "run msame failed"; return $ret_run_failed; }

    echo "ais_infer analyser"
    cmd="$PYTHON_COMMAND $CUR_PATH/analyser.py --mode times --summary_path  $CACHE_PATH/asumary.json"
    $cmd || { echo "cmd:$cmd analyse times ais_infer failed"; return $ret_run_failed; }

    echo "msame analyser"
    cmd="$PYTHON_COMMAND $CUR_PATH/analyser.py --mode times --summary_path  $CACHE_PATH/msumary.json"
    $cmd || { echo "cmd:$cmd analyse times msame failed"; return $ret_run_failed; }

    return $ret_ok
}

main "$@"
exit $?
