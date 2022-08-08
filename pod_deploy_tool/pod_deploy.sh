#!/bin/bash

function remove_single_quote() {
    local tmp=$*
    local var=${tmp#?}
    var=${var%?}
    echo "${var}"
}

function use_help() {
    echo "--work_dir atlas edge work dir, eg:/usr/local/AtlasEdge"
    echo "--operate  pod operate type, create ore delete"
    echo "--pod_file absolute path of pod yaml file"
}

function main() {
    local para=( "$@" )
    local args
    args=$(getopt -o h -a -l help,"work_dir:,operate:,pod_file:" -n "pod_deploy.sh" -- "$@")
    if [[ $? != 0 ]];then
        echo "parse para failed, try '--help' for more information"
        return 1
    fi

    set -- ${args}
    while true ; do
        case "$1" in
        --work_dir | -work_dir)
          atlas_edge_work=$(remove_single_quote "$2")
          shift 2
          ;;
        -h | --help)
          use_help
          return 1
          ;;
        --)
          shift
          break
          ;;
        *)
          shift 2
          ;;
        esac
    done

    if [[ ! -e ${atlas_edge_work}/edge_work_dir/edge_om/env_profile.sh ]]; then
        echo "work_dir ${atlas_edge_work} invalid, try '--help' for more information"
        return 1
    fi

    source "${atlas_edge_work}"/edge_work_dir/edge_om/env_profile.sh
    python3 pod_deploy.py "${para[@]}"
}

main "$@"
exit $?