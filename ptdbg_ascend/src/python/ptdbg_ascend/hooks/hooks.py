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

import json
import os
import random
import stat

import numpy as np
import torch
import torch_npu

from ..common.utils import check_file_or_directory_path, print_error_log, CompareException, Const


def set_dump_path(fpath=None):
    if fpath is None:
        return
    real_path = os.path.realpath(fpath)
    if os.path.isdir(real_path):
        print_error_log("set_dump_path '{}' error, please set a valid filename.".format(real_path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    check_file_or_directory_path(os.path.dirname(real_path), True)
    if os.path.exists(real_path):
        os.remove(real_path)
    os.environ["DUMP_PATH"] = real_path


def get_dump_path():
    assert "DUMP_PATH" in os.environ, "Please set dump path for ptdbg_ascend tools."
    return os.environ.get("DUMP_PATH")


def set_dump_switch(switch=None):
    if switch is None:
        return
    assert switch in ["ON", "OFF"], "Please set dump switch with 'ON' or 'OFF'."
    os.environ["PYTORCH_DUMP_SWITCH"] = switch


def get_dump_switch():
    assert "PYTORCH_DUMP_SWITCH" in os.environ, "Please set dump switch for ptdbg_ascend tools."
    switch = os.environ.get("PYTORCH_DUMP_SWITCH")
    if switch == "ON":
        return True
    else:
        return False


def set_seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dump_tensor(x, prefix, dump_mode):
    if "DUMP_PATH" not in os.environ:
        return
    if get_dump_switch():
        dump_process(x, prefix, dump_mode)


def dump_process(x, prefix, dump_mode):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_process(item, "{}.{}".format(prefix, i), dump_mode)
    elif isinstance(x, torch.Tensor):
        if len(x.shape) == 0 or not x.is_floating_point():
            return

        f = os.fdopen(os.open(get_dump_path(), os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a")
        summery_data = []
        if dump_mode == Const.DUMP_MODE.get("SUMMERY"):
            tensor_sum = torch._C._VariableFunctionsClass.sum(x).cpu().detach().float().numpy().tolist()
            tensor_mean = torch._C._VariableFunctionsClass.mean(x).cpu().detach().float().numpy().tolist()
            saved_tensor = x.contiguous().view(-1)[:Const.SUMMERY_DATA_NUMS].cpu().detach().float().numpy().tolist()
            summery_data.extend([tensor_sum, tensor_mean])
        elif dump_mode == Const.DUMP_MODE.get("SAMPLE"):
            np.random.seed(int(os.environ['PYTHONHASHSEED']))
            sample_ratio = x.shape[0] // Const.SUMMERY_DATA_NUMS if x.shape[0] >= Const.SUMMERY_DATA_NUMS else x.shape[0]
            sample_index = np.sort(np.random.choice(x.shape[0], sample_ratio, replace='False'))
            saved_tensor = x.contiguous()[sample_index].view(-1).cpu().detach().float().numpy().tolist()
        elif dump_mode == Const.DUMP_MODE.get("ALL"):
            saved_tensor = x.contiguous().view(-1).cpu().detach().float().numpy().tolist()
        else:
            print_error_log("dump_mode is invalid, Please set dump mode in [1, 2, 3],"
                            " 1: SUMMARY mode, 2: SAMPLE mode, 3: ALL mode")
            f.close()
            raise CompareException(CompareException.INVALID_DUMP_MODE)

        json.dump([prefix, dump_mode, saved_tensor, str(x.dtype), tuple(x.shape), summery_data], f)
        f.write('\n')
        f.close()


def wrap_acc_cmp_hook(name, **kwargs):
    dump_mode = kwargs.get('dump_mode', 1)

    def acc_cmp_hook(module, in_feat, out_feat):
        name_template = f"{name}" + "_{}"
        dump_tensor(in_feat, name_template.format("input"), dump_mode)
        dump_tensor(out_feat, name_template.format("output"), dump_mode)

    return acc_cmp_hook


def wrap_checkoverflow_hook(name, **kwargs):
    def checkoverflow_hook(module, in_feat, out_feat):
        module_name = name
        module.has_overflow = torch_npu._C._check_overflow_npu()
        if module.has_overflow:
            raise ValueError("[check overflow]:module name :'{}' is overflow!".format(module_name))

    return checkoverflow_hook
