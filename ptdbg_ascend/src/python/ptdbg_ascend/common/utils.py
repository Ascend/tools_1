#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import collections
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
if not torch.cuda.is_available():
    import torch_npu


device = collections.namedtuple('device', ['type', 'index'])


class Const:
    """
    Class for const
    """
    MODEL_TYPE = ['.onnx', '.pb', '.om']
    DIM_PATTERN = r"^(-?[0-9]+)(,-?[0-9]+)*"
    SEMICOLON = ";"
    COLON = ":"
    EQUAL = "="
    COMMA = ","
    DOT = "."
    DUMP_MODE = {"SUMMERY": 1, "SAMPLE": 2, "ALL": 3}
    DUMP_SCOPE = {"ALL": 1, "LIST": 2, "RANGE": 3, "STACK": 4}
    SUMMERY_DATA_NUMS = 256
    FLOAT_EPSILON = np.finfo(float).eps
    NAN = 'NaN'


class VersionCheck:
    """
    Class for TorchVersion
    """
    V1_8 = "1.8"
    V1_11 = "1.11"

    @staticmethod
    def check_torch_version(version):
        torch_version = torch.__version__
        if torch_version.startswith(version):
            return True
        else:
            return False


class CompareException(Exception):
    """
    Class for Accuracy Compare Exception
    """
    NONE_ERROR = 0
    INVALID_PATH_ERROR = 1
    OPEN_FILE_ERROR = 2
    CLOSE_FILE_ERROR = 3
    READ_FILE_ERROR = 4
    WRITE_FILE_ERROR = 5
    INVALID_FILE_ERROR = 6
    PERMISSION_ERROR = 7
    INDEX_OUT_OF_BOUNDS_ERROR = 8
    NO_DUMP_FILE_ERROR = 9
    INVALID_DATA_ERROR = 10
    INVALID_PARAM_ERROR = 11
    INVALID_DUMP_MODE = 12
    INVALID_DUMP_FILE = 13
    UNKNOWN_ERROR = 14

    def __init__(self, code, error_info: str = ""):
        super(CompareException, self).__init__()
        self.code = code
        self.error_info = error_info


def _print_log(level, msg):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getgid()
    print(current_time + "(" + str(pid) + ")-[" + level + "]" + msg)
    sys.stdout.flush()


def print_info_log(info_msg):
    """
    Function Description:
        print info log.
    Parameter:
        info_msg: the info message.
    """
    _print_log("INFO", info_msg)


def print_error_log(error_msg):
    """
    Function Description:
        print error log.
    Parameter:
        error_msg: the error message.
    """
    _print_log("ERROR", error_msg)


def print_warn_log(warn_msg):
    """
    Function Description:
        print warn log.
    Parameter:
        warn_msg: the warning message.
    """
    _print_log("WARNING", warn_msg)


def check_file_or_directory_path(path, isdir=False):
    """
    Function Description:
        check whether the path is valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
    Exception Description:
        when invalid data throw exception
    """
    if isdir:
        if not os.path.exists(path):
            print_error_log('The path {} is not exist. Please check the path'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory. Please check the path'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.access(path, os.W_OK):
            print_error_log(
                'The path{} does not have permission to write. Please check the path permission'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('The path {} is not a file. Please check the path'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log(
            'The path{} does not have permission to read. Please check the path permission'.format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)


def get_dump_data_path(dump_dir):
    """
    Function Description:
        traverse directories and obtain the absolute path of dump data
    Parameter:
        dump_dir: dump data directory
    Return Value:
        dump data path,file is exist or file is not exist
    """
    dump_data_path = None
    file_is_exist = False

    check_file_or_directory_path(dump_dir, True)
    for dir_path, sub_paths, files in os.walk(dump_dir):
        if len(files) != 0:
            dump_data_path = dir_path
            file_is_exist = True
            break
        dump_data_path = dir_path
    return dump_data_path, file_is_exist


def create_directory(dir_path):
    """
    Function Description:
        creating a directory with specified permissions
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode=0o700)
        except OSError as ex:
            print_error_log(
                'Failed to create {}.Please check the path permission or disk space .{}'.format(dir_path, str(ex)))
            raise CompareException(CompareException.INVALID_PATH_ERROR)


def execute_command(cmd):
    """
    Function Description:
        run the following command
    Parameter:
        cmd: command
    Exception Description:
        when invalid command throw exception
    """
    print_info_log('Execute command:%s' % cmd)
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.strip()
        if line:
            print(line)
    if process.returncode != 0:
        print_error_log('Failed to execute command:%s' % " ".join(cmd))
        raise CompareException(CompareException.INVALID_DATA_ERROR)


def save_numpy_data(file_path, data):
    """
    save_numpy_data
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, data)


def parse_arg_value(values):
    """
    parse dynamic arg value of atc cmdline
    """
    value_list = []
    for item in values.split(Const.SEMICOLON):
        value_list.append(parse_value_by_comma(item))
    return value_list


def parse_value_by_comma(value):
    """
    parse value by comma, like '1,2,4,8'
    """
    value_list = []
    value_str_list = value.split(Const.COMMA)
    for value_str in value_str_list:
        value_str = value_str.strip()
        if value_str.isdigit() or value_str == '-1':
            value_list.append(int(value_str))
        else:
            print_error_log("please check your input shape.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return value_list


def get_data_len_by_shape(shape):
    data_len = 1
    for item in shape:
        if item is -1:
            print_error_log("please check your input shape, one dim in shape is -1.")
            return -1
        data_len = data_len * item
    return data_len


def add_time_as_suffix(name):
    return '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def get_time():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def format_value(value):
    return '{:.6f}'.format(value)


def torch_device_guard(func):
    # Parse args/kwargs from namedtuple(torch.device) to matched torch.device objects
    def wrapper(*args, **kwargs):
        if args:
            args_list = list(args)
            for index, arg in enumerate(args_list):
                if isinstance(arg, tuple) and "type='npu'" in str(arg):
                    args_list[index] = torch_npu.new_device(type=torch_npu.npu.native_device, index=arg.index)
                    break
            args = tuple(args_list)
        if kwargs and isinstance(kwargs.get("device"), tuple):
            namedtuple_device = kwargs.get("device")
            if "type='npu'" in str(namedtuple_device):
                kwargs['device'] = torch_npu.new_device(type=torch_npu.npu.native_device, index=namedtuple_device.index)
        return func(*args, **kwargs)
    return wrapper
