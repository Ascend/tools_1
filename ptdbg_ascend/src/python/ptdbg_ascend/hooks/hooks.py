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
    print_warn_log, CompareException, Const, get_time, print_info_log
from .backward import Backward


class DumpUtil(object):
    dump_data_dir = None
    dump_path = None
    dump_switch = None
    dump_switch_mode = Const.DUMP_SCOPE.get("ALL")
    dump_switch_scope = []
    dump_init_enable = False
    real_overflow_dump_times = 0

    @staticmethod
    def set_dump_path(save_path):
        DumpUtil.dump_path = save_path
        DumpUtil.dump_init_enable = True

    @staticmethod
    def set_dump_switch(switch, mode, scope):
        DumpUtil.dump_switch = switch
        DumpUtil.dump_switch_mode = mode
        DumpUtil.dump_init_enable = True
        DumpUtil.dump_switch_scope = scope

    @staticmethod
    def check_switch_scope(name_prefix):
        if DumpUtil.dump_switch_mode == Const.DUMP_SCOPE.get("ALL"):
            return True
        elif DumpUtil.dump_switch_mode == Const.DUMP_SCOPE.get("LIST"):
            for item in DumpUtil.dump_switch_scope:
                if name_prefix.startswith(item):
                    return True
        elif DumpUtil.dump_switch_mode == Const.DUMP_SCOPE.get("RANGE"):
            start = int(DumpUtil.dump_switch_scope[0].split('_', 1)[0])
            end = int(DumpUtil.dump_switch_scope[1].split('_', 1)[0])
            curr = int(name_prefix.split('_', 1)[0])
            if start <= curr <= end:
                return True
        elif DumpUtil.dump_switch_mode == Const.DUMP_SCOPE.get("STACK"):
            if len(DumpUtil.dump_switch_scope) == 0:
                return True
            elif len(DumpUtil.dump_switch_scope) == 1:
                if name_prefix.startswith(DumpUtil.dump_switch_scope[0]):
                    return True
            elif len(DumpUtil.dump_switch_scope) == 2:
                start = int(DumpUtil.dump_switch_scope[0].split('_', 1)[0])
                end = int(DumpUtil.dump_switch_scope[1].split('_', 1)[0])
                curr = int(name_prefix.split('_', 1)[0])
                if start <= curr <= end:
                    return True
            else:
                print_error_log("dump scope is invalid, Please set the scope mode in"
                                " set_dump_switch with [1, 2, 3, 4],1: ALL, 2: List, 3: ARRANGE, 4: STACK !")

        return False

    @staticmethod
    def get_dump_path():
        if DumpUtil.dump_path:
            return DumpUtil.dump_path

        if DumpUtil.dump_switch_mode == Const.DUMP_SCOPE.get("ALL"):
            raise RuntimeError("get_dump_path: the file path is empty,"
                               " you must use set_dump_path to set a valid dump path!!!")
        else:
            dir_path = os.path.realpath("./")
            dump_file_name = "scope_dump_{}_{}_{}.pkl".format(
                DumpUtil.dump_switch_mode, DumpUtil.dump_switch_scope[0], get_time())
            DumpUtil.dump_path = os.path.join(dir_path, dump_file_name)
            return DumpUtil.dump_path

    @staticmethod
    def get_dump_switch():
        if DumpUtil.dump_switch is None:
            return False

        if DumpUtil.dump_switch == "ON":
            return True
        else:
            return False

    @staticmethod
    def inc_overflow_dump_times():
        DumpUtil.real_overflow_dump_times += 1

    @staticmethod
    def check_overflow_dump_times(need_dump_times):
        if DumpUtil.real_overflow_dump_times < need_dump_times:
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


def set_dump_switch(switch, mode=1, scope=[]):
    assert switch in ["ON", "OFF"], "Please set dump switch with 'ON' or 'OFF'."
    if mode == Const.DUMP_SCOPE.get("RANGE"):
        assert len(scope) == 2, "set_dump_switch, scope param set invalid, it's must be [start, end]."
    if mode == Const.DUMP_SCOPE.get("LIST"):
        assert len(scope) != 0, "set_dump_switch, scope param set invalid, it's should not be an empty list."
    if mode == Const.DUMP_SCOPE.get("STACK"):
        assert len(scope) <= 2, "set_dump_switch, scope param set invalid, it's must be [start, end] or []."
    DumpUtil.set_dump_switch(switch, mode=mode, scope=scope)

def dump_tensor(x, prefix, dump_step):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, "{}.{}".format(prefix, i), dump_step)
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 or len(x.shape) == 0 or not x.is_floating_point():
            return

        with os.fdopen(os.open(DumpUtil.get_dump_path(), os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a") as f:
            summery_data = []
            if 1 <= dump_step <= Const.DUMP_RATIO_MAX:
                tensor_max = torch._C._VariableFunctionsClass.max(x).cpu().detach().float().numpy().tolist()
                tensor_min = torch._C._VariableFunctionsClass.min(x).cpu().detach().float().numpy().tolist()
                tensor_mean = torch._C._VariableFunctionsClass.mean(x).cpu().detach().float().numpy().tolist()
                tensor_len = torch._C._VariableFunctionsClass.numel(x)
                if tensor_len <= Const.SUMMERY_DATA_NUMS * 2:
                    saved_tensor = x.contiguous().view(-1)[:Const.SUMMERY_DATA_NUMS*2].cpu().detach().float().numpy()
                else:
                    saved_tensor_head = x.contiguous().view(-1)[
                                        :Const.SUMMERY_DATA_NUMS].cpu().detach().float().numpy()
                    saved_tensor_body = []
                    if dump_step != 0:
                        saved_tensor_body = x.contiguous().view(-1)[Const.SUMMERY_DATA_NUMS: (-1 * Const.SUMMERY_DATA_NUMS): dump_step]\
                            .cpu().detach().float().numpy()
                    saved_tensor_tail = x.contiguous().view(-1)[
                                        (-1 * Const.SUMMERY_DATA_NUMS):].cpu().detach().float().numpy()
                    saved_tensor = np.append(np.append(saved_tensor_head, saved_tensor_body), saved_tensor_tail)
                summery_data.extend([tensor_max, tensor_min, tensor_mean])
            else:
                print_error_log("dump_ratio is invalid, Please set dump mode in [1~100], indicate 1% ~ 100%.")
                f.close()
                raise CompareException(CompareException.INVALID_DUMP_RATIO)

            output_path = os.path.join(DumpUtil.dump_data_dir, f'{prefix}.npy')
            np.save(output_path, saved_tensor)
            json.dump([prefix, dump_step, [], str(x.dtype), tuple(x.shape), summery_data], f)

            f.write('\n')


def _dump_tensor_completely(x, prefix, dump_file_name):
    if "stack_info" in prefix:
        with os.fdopen(os.open(dump_file_name, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "a") as f:
            json.dump([prefix, x], f)
            f.write('\n')
        return

    dump_flag = Const.DUMP_RATIO_MAX + 1
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            _dump_tensor_completely(item, "{}.{}".format(prefix, i), dump_file_name)
    else:
        with os.fdopen(os.open(dump_file_name, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "a") as f:
            if isinstance(x, torch.Tensor) and x.numel() != 0:
                output_path = os.path.join(DumpUtil.dump_data_dir, f'{prefix}.npy')
                save_tensor = x.contiguous().view(-1).cpu().detach().float().numpy()
                np.save(output_path, save_tensor)
                json.dump([prefix, dump_flag, [], str(x.dtype), tuple(x.shape)], f)
            f.write('\n')


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dump_data_dir(dump_file_name):
    dump_path, file_name = os.path.split(os.path.realpath(dump_file_name))
    name_body, name_extension = os.path.splitext(file_name)
    output_dir = os.path.join(dump_path, f"{name_body}_{get_time()}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir

def dump_acc_cmp(name, in_feat, out_feat, dump_step):
    dump_file = DumpUtil.get_dump_path()
    if DumpUtil.get_dump_switch():
        if DumpUtil.dump_init_enable:
            dump_acc_cmp.call_number = 0
            DumpUtil.dump_init_enable = False
            DumpUtil.dump_data_dir = make_dump_data_dir(dump_file)
        else:
            dump_acc_cmp.call_number = dump_acc_cmp.call_number + 1

        name_prefix = f"{dump_acc_cmp.call_number}_{name}"
        if DumpUtil.dump_switch_mode == Const.DUMP_SCOPE.get("ALL"):
            name_template = f"{name_prefix}" + "_{}"
            dump_tensor(in_feat, name_template.format("input"), dump_step)
            dump_tensor(out_feat, name_template.format("output"), dump_step)
        elif DumpUtil.check_switch_scope(name_prefix):
            name_template = f"{name_prefix}" + "_{}"
            stack_str = []
            for (_, path, line, func, code, _) in inspect.stack()[3:]:
                stack_line = [path, str(line), func, code[0].strip()]
                stack_str.append(stack_line)
            _dump_tensor_completely(stack_str, name_template.format("stack_info"), dump_file)
            if DumpUtil.dump_switch_mode != Const.DUMP_SCOPE.get("STACK"):
                dump_tensor(in_feat, name_template.format("input"), dump_step)
                dump_tensor(out_feat, name_template.format("output"), dump_step)


def acc_cmp_dump(name, **kwargs):
    dump_step = kwargs.get('dump_step', 1)
    pid = kwargs.get('pid')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(module, in_feat, out_feat):
        if pid == os.getpid():
            dump_acc_cmp(name, in_feat, out_feat, dump_step)

    return acc_cmp_hook


def dump_overflow(module_name, stack_str, in_feat, out_feat, dump_file):
    name_template = f"{module_name}" + "_{}"
    DumpUtil.dump_data_dir = make_dump_data_dir(dump_file)
    _dump_tensor_completely(stack_str, name_template.format("stack_info"), dump_file)
    _dump_tensor_completely(in_feat, name_template.format("input"), dump_file)
    _dump_tensor_completely(out_feat, name_template.format("output"), dump_file)


def overflow_check(name, **kwargs):
    overflow_nums = kwargs.get('overflow_nums', 1)
    pid = kwargs.get('pid')
    dump_mode = kwargs.get('dump_mode', "api")
    dump_config = kwargs.get('dump_config')
    backward_obj = Backward()
    torch.autograd.backward = backward_obj.backward
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
        if not module.has_overflow and "forward" in module_name:
            del module.input_args
            del module.input_kwargs
        if module.has_overflow and DumpUtil.check_overflow_dump_times(overflow_nums):
            DumpUtil.inc_overflow_dump_times()
            dump_file_name = "Overflow_info_{}.pkl".format(get_time())
            stack_str = []
            for (_, path, line, func, code, _) in inspect.stack()[3:]:
                stack_line = [path, str(line), func, code[0].strip()]
                stack_str.append(stack_line)
            dump_overflow(module_name, stack_str, in_feat, out_feat, dump_file_name)
            # clear overflow flag for the next check
            torch_npu._C._clear_overflow_npu()
            if dump_mode == "acl":
                acl_dump(module, module_name)
            print_warn_log("[overflow {} times]: module name :'{}' is overflow and dump file is saved in '{}'."
                           .format(DumpUtil.real_overflow_dump_times, module_name, os.path.realpath(dump_file_name)))
            if not DumpUtil.check_overflow_dump_times(overflow_nums):
                raise ValueError("[overflow {} times]: dump file is saved in '{}'."
                                 .format(DumpUtil.real_overflow_dump_times, os.path.realpath(dump_file_name)))
        else:
            print_info_log("No overflow/underflow occurs on the {} operator".format(module_name))

    def acl_dump(module, module_name):
        if "forward" in module_name:
            forward_acl_dump(module, module_name)
        if "backward" in module_name:
            backward_acl_dump()

    def backward_acl_dump():
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(dump_config)
        torch_npu.npu.synchronize()
        torch.autograd.backward(backward_obj.tensors, backward_obj.gradient, backward_obj.retain_graph,
                                backward_obj.create_graph, inputs=backward_obj.inputs)
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
        print_info_log("Dump backward op file.")
        raise ValueError("[Acl backward only support one time, will stop when detecct backward overflow]")

    def forward_acl_dump(module, module_name):
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(dump_config)
        torch_npu.npu.synchronize()
        module.forward(*module.input_args, **module.input_kwargs)
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
        del module.input_args
        del module.input_kwargs
        print_info_log("Dump %s op file." % module_name)

    return overflowcheck_hook
