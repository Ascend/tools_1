#!/bin/bash
export PYTHON_COMMAND=python3.7
export TRAIN_DATA_FILE=/home/datasets/criteo/mini_demo.txt

export RANK_SIZE=8
export DEVICE_NUM=8

# need if rank_size > 1
export RANK_TABLE_FILE=/home/lcm/tool/rank_table_16p_62_64.json
# cluster need for node info
#export NODEINFO_FILE=/home/lcm/tool/ssh64_66.json

