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

import inspect
import json
import os
import random
import stat

import numpy as np
import torch

if not torch.cuda.is_available():
    import torch_npu

from ..common.utils import check_file_or_directory_path, print_error_log, \
    print_warn_log, CompareException, Const, get_time


class DumpUtil(object):
    dump_path = None
    dump_switch = None
    dump_init_enable = False

    @staticmethod
    def set_dump_path(save_path):
        DumpUtil.dump_path = save_path
        DumpUtil.dump_init_enable = True

    @staticmethod
    def set_dump_switch(switch):
        DumpUtil.dump_switch = switch
        DumpUtil.dump_init_enable = True

    @staticmethod
    def get_dump_path():
        return DumpUtil.dump_path

    @staticmethod
    def get_dump_switch():
        if DumpUtil.dump_switch is None:
            return False

        if DumpUtil.dump_switch == "ON":
            return True
        else:
            return False


def set_dump_path(fpath=None):
    if fpath is None:
        raise RuntimeError("set_dump_path '{}' error, please set a valid filename".format(fpath))
        return
    real_path = os.path.realpath(fpath)
    if os.path.isdir(real_path):
        print_error_log("set_dump_path '{}' error, please set a valid filename.".format(real_path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    check_file_or_directory_path(os.path.dirname(real_path), True)
    if os.path.exists(real_path):
        os.remove(real_path)
    DumpUtil.set_dump_path(real_path)


def set_dump_switch(switch=None):
    if switch is None:
        return
    assert switch in ["ON", "OFF"], "Please set dump switch with 'ON' or 'OFF'."
    DumpUtil.set_dump_switch(switch)


def dump_tensor(x, prefix, dump_mode):
    if DumpUtil.get_dump_path() is None:
        return
    if DumpUtil.get_dump_switch():
        dump_process(x, prefix, dump_mode)


def dump_process(x, prefix="", dump_mode=1):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_process(item, prefix="{}.{}".format(prefix, i), dump_mode=1)
    elif isinstance(x, torch.Tensor):
        if len(x.shape) == 0 or not x.is_floating_point():
            return

        if DumpUtil.dump_init_enable:
            dump_process.call_number = 0
            DumpUtil.dump_init_enable = False
        else:
            dump_process.call_number = dump_process.call_number + 1
        prefix = f"{dump_process.call_number}_{prefix}"

        with os.fdopen(os.open(DumpUtil.get_dump_path(), os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a") as f:
            summery_data = []
            if dump_mode == Const.DUMP_MODE.get("SUMMERY"):
                tensor_max = torch._C._VariableFunctionsClass.max(x).cpu().detach().float().numpy().tolist()
                tensor_min = torch._C._VariableFunctionsClass.min(x).cpu().detach().float().numpy().tolist()
                tensor_mean = torch._C._VariableFunctionsClass.mean(x).cpu().detach().float().numpy().tolist()
                tensor_len = torch._C._VariableFunctionsClass.numel(x)
                if tensor_len <= Const.SUMMERY_DATA_NUMS * 2:
                    saved_tensor = x.contiguous().view(-1)[:Const.SUMMERY_DATA_NUMS*2].cpu().detach().float().numpy().tolist()
                else:
                    saved_tensor_head = x.contiguous().view(-1)[
                                        :Const.SUMMERY_DATA_NUMS].cpu().detach().float().numpy().tolist()
                    saved_tensor_tail = x.contiguous().view(-1)[
                                        Const.SUMMERY_DATA_NUMS:].cpu().detach().float().numpy().tolist()
                    saved_tensor = saved_tensor_head + saved_tensor_tail
                summery_data.extend([tensor_max, tensor_min, tensor_mean])
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


def _dump_tensor_for_overflow(x, dump_file_name, prefix=""):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            _dump_tensor_for_overflow(item, dump_file_name, prefix="{}.{}".format(prefix, i))
    else:
        with os.fdopen(os.open(dump_file_name, os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a") as f:
            if isinstance(x, torch.Tensor):
                save_tensor = x.contiguous().view(-1).cpu().detach().float().numpy().tolist()
                json.dump([prefix, save_tensor, str(x.dtype), tuple(x.shape)], f)
            else:
                json.dump([prefix, x], f)
            f.write('\n')


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def acc_cmp_dump(name, **kwargs):
    dump_mode = kwargs.get('dump_mode', 1)
    pid = kwargs.get('pid')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(module, in_feat, out_feat):
        if pid == os.getpid():
            name_template = f"{name}" + "_{}"
            dump_tensor(in_feat, name_template.format("input"), dump_mode)
            dump_tensor(out_feat, name_template.format("output"), dump_mode)

    return acc_cmp_hook


def overflow_check(name, **kwargs):
    pid = kwargs.get('pid')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def overflowcheck_hook(module, in_feat, out_feat):
        if pid != os.getpid():
            return
        if torch.cuda.is_available():
            print_warn_log("Overflow detection is not supported in the GPU environment.")
            return
        module_name = name
        module.has_overflow = torch_npu._C._check_overflow_npu()
        if module.has_overflow:
            name_template = f"{name}" + "_{}"
            dump_file_name = "Overflow_info_{}.pkl".format(get_time())
            stack_str = [str(_) for _ in inspect.stack()[3:]]
            _dump_tensor_for_overflow(stack_str, dump_file_name, name_template.format("stack_info"))
            _dump_tensor_for_overflow(in_feat, dump_file_name, name_template.format("input"))
            _dump_tensor_for_overflow(out_feat, dump_file_name, name_template.format("output"))
            raise ValueError("[check overflow]: module name :'{}' is overflow and dump file is saved in '{}'.".format(
                module_name, os.path.realpath(dump_file_name)))

    return overflowcheck_hook
