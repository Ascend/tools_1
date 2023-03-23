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
    print_warn_log, CompareException, Const, get_time, print_info_log, modify_dump_path
from .backward import Backward

DumpCount = 0
init_status = False


class DumpUtil(object):
    dump_data_dir = None
    dump_path = None
    dump_switch = None
    dump_switch_mode = Const.ALL
    dump_switch_scope = []
    dump_init_enable = False
    dump_api_list = []

    @staticmethod
    def set_dump_path(save_path):
        DumpUtil.dump_path = save_path
        DumpUtil.dump_init_enable = True

    @staticmethod
    def set_dump_switch(switch, mode, scope, api_list):
        DumpUtil.dump_switch = switch
        DumpUtil.dump_switch_mode = mode
        DumpUtil.dump_init_enable = True
        DumpUtil.dump_switch_scope = scope
        DumpUtil.dump_api_list = [api.lower() for api in api_list]

    def check_list_or_acl_mode(name_prefix):
        global DumpCount
        for item in DumpUtil.dump_switch_scope:
            if name_prefix.startswith(item):
                DumpCount = DumpCount + 1
                return True

    def check_range_mode(name_prefix):
        start = int(DumpUtil.dump_switch_scope[0].split('_', 1)[0])
        end = int(DumpUtil.dump_switch_scope[1].split('_', 1)[0])
        curr = int(name_prefix.split('_', 1)[0])
        return start <= curr <= end

    def check_stack_mode(name_prefix):
        if len(DumpUtil.dump_switch_scope) == 0:
            return True
        elif len(DumpUtil.dump_switch_scope) == 1:
            return name_prefix.startswith(DumpUtil.dump_switch_scope[0])
        elif len(DumpUtil.dump_switch_scope) == 2:
            return DumpUtil.check_range_mode(name_prefix)
        else:
            print_error_log("dump scope is invalid, Please set the scope mode in"
                            " set_dump_switch with 'all', 'list', 'range', 'stack', 'acl', 'api_list'!")
        return False

    check_mapper = {
        Const.LIST: check_list_or_acl_mode,
        Const.ACL: check_list_or_acl_mode,
        Const.RANGE: check_range_mode,
        Const.STACK: check_stack_mode
    }

    @staticmethod
    def check_switch_scope(name_prefix):
        if DumpUtil.dump_switch_mode in DumpUtil.check_mapper:
            check_func = DumpUtil.check_mapper[DumpUtil.dump_switch_mode]
            return check_func(name_prefix)
        return False

    @staticmethod
    def get_dump_path():
        if DumpUtil.dump_path:
            return DumpUtil.dump_path

        if DumpUtil.dump_switch_mode == Const.ALL:
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
        return DumpUtil.dump_switch == "ON"


class OverFlowUtil(object):
    overflow_check_switch = None
    real_overflow_dump_times = 0

    @staticmethod
    def set_overflow_check_switch(switch):
        OverFlowUtil.overflow_check_switch = switch

    @staticmethod
    def get_overflow_check_switch():
        if OverFlowUtil.overflow_check_switch is None:
            return True
        return OverFlowUtil.overflow_check_switch == "ON"

    @staticmethod
    def inc_overflow_dump_times():
        OverFlowUtil.real_overflow_dump_times += 1

    @staticmethod
    def check_overflow_dump_times(need_dump_times):
        return OverFlowUtil.real_overflow_dump_times < need_dump_times


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


def set_dump_switch(switch, mode=Const.ALL, scope=[], api_list=[]):
    global DumpCount
    assert switch in ["ON", "OFF"], "Please set dump switch with 'ON' or 'OFF'."
    if mode == Const.LIST and switch == "ON":
        DumpCount = 0
    if mode == Const.LIST and switch == "OFF":
        print_info_log("The number of matched dump is {}".format(DumpCount))
    if mode == Const.RANGE:
        assert len(scope) == 2, "set_dump_switch, scope param set invalid, it's must be [start, end]."
    if mode == Const.LIST:
        assert len(scope) != 0, "set_dump_switch, scope param set invalid, it's should not be an empty list."
    if mode == Const.STACK:
        assert len(scope) <= 2, "set_dump_switch, scope param set invalid, it's must be [start, end] or []."
    DumpUtil.set_dump_switch(switch, mode=mode, scope=scope, api_list=api_list)


def set_overflow_check_switch(switch):
    assert switch in ["ON", "OFF"], "Please set overflow switch with 'ON' or 'OFF'."
    OverFlowUtil.set_overflow_check_switch(switch)


def dump_tensor(x, prefix, dump_step, dump_file_name):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, "{}.{}".format(prefix, i), dump_step, dump_file_name)
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 or len(x.shape) == 0 or not x.is_floating_point():
            return

        with os.fdopen(os.open(dump_file_name, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
                       "a") as f:
            summery_data = []

            tensor_max = torch._C._VariableFunctionsClass.max(x).cpu().detach().float().numpy().tolist()
            tensor_min = torch._C._VariableFunctionsClass.min(x).cpu().detach().float().numpy().tolist()
            tensor_mean = torch._C._VariableFunctionsClass.mean(x).cpu().detach().float().numpy().tolist()
            dump_flag = Const.DUMP_RATIO_MAX + 1
            saved_tensor = x.contiguous().cpu().detach().numpy()
            summery_data.extend([tensor_max, tensor_min, tensor_mean])

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
                save_tensor = x.contiguous().cpu().detach().numpy()
                np.save(output_path, save_tensor)
                json.dump([prefix, dump_flag, [], str(x.dtype), tuple(x.shape)], f)
            f.write('\n')


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_process_rank(model):
    print("Rank id is not provided. Trying to get the rank id of the model.")
    try:
        device = next(model.parameters()).device 
    except StopIteration:
        print('There is no parameter in the model. Fail to get rank id.')
        return 0
    if device.type == 'cpu':
        print("Warning: the debugger is unable to get the rank id. "
            "This may cause the dumpped data to be corrupted in the "
            "case of DDP. Transfer the model to npu or gpu before "
            "register_hook() to avoid this warning.")
        return 0
    else:
        return device.index

def make_dump_dirs(rank, pid):
    if DumpUtil.dump_path is not None:
        dump_root_dir, dump_file_name = os.path.split(DumpUtil.dump_path)
        dump_file_name_body, _ = os.path.splitext(dump_file_name)
    else:
        dump_root_dir, dump_file_name, dump_file_name_body = './', 'dummy.pkl', ''
    time = get_time()
    time_dir = os.path.join(dump_root_dir, dump_file_name_body + '_' + str(time))
    if rank == 0 and not os.path.exists(time_dir): # add rank==0 to prevent repeated mkdir
        os.mkdir(time_dir)
    while not os.path.exists(time_dir): # wait for rank 0 process to create timedir
        pass 
    rank_dir = os.path.join(time_dir, 'rank' + str(rank))
    if not os.path.exists(rank_dir):
        os.mkdir(rank_dir)
    pid_dir = os.path.join(rank_dir, 'pid' + str(pid))
    if not os.path.exists(pid_dir):
        os.mkdir(pid_dir)
    DumpUtil.dump_dir = pid_dir 
    DumpUtil.set_dump_path(os.path.join(pid_dir, dump_file_name))

def make_dump_data_dir(dump_file_name):
    dump_path, file_name = os.path.split(os.path.realpath(dump_file_name))
    name_body, name_extension = os.path.splitext(file_name)
    output_dir = os.path.join(dump_path, f"{name_body}_{get_time()}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


def _set_dump_switch4api_list(name):
    if DumpUtil.dump_api_list:
        api_name = name.split("_")[1].lower()
        if api_name in DumpUtil.dump_api_list and DumpUtil.dump_switch == "ON":
            DumpUtil.dump_switch = "OFF"


def dump_stack_info(name_template, dump_file):
    stack_str = []
    for (_, path, line, func, code, _) in inspect.stack()[3:]:
        stack_line = [path, str(line), func, code[0].strip()]
        stack_str.append(stack_line)
    _dump_tensor_completely(stack_str, name_template.format("stack_info"), dump_file)


def dump_acc_cmp(name, in_feat, out_feat, dump_step, moudle):
    dump_file = DumpUtil.get_dump_path()
    _set_dump_switch4api_list(name)
    if DumpUtil.dump_switch_mode == Const.API_STACK:
        dump_file = modify_dump_path(dump_file)

    if DumpUtil.get_dump_switch():
        if DumpUtil.dump_init_enable:
            dump_acc_cmp.call_number = 0
            DumpUtil.dump_init_enable = False
            DumpUtil.dump_data_dir = make_dump_data_dir(dump_file) \
                if DumpUtil.dump_switch_mode != Const.STACK else ""
        else:
            dump_acc_cmp.call_number = dump_acc_cmp.call_number + 1

        name_prefix = f"{dump_acc_cmp.call_number}_{name}"
        name_template = f"{name_prefix}" + "_{}"
        if DumpUtil.dump_switch_mode in [Const.ALL, Const.API_LIST]:
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file)
        elif DumpUtil.dump_switch_mode == Const.API_STACK:
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file)
            dump_stack_info(name_template, dump_file)
        elif DumpUtil.check_switch_scope(name_prefix):
            dump_stack_info(name_template, dump_file)
            if DumpUtil.dump_switch_mode == Const.ACL:
                acl_dump(moudle, name)
            elif DumpUtil.dump_switch_mode != Const.STACK:
                dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file)


def acl_dump(module, module_name):
    if "forward" in module_name:
        forward_acl_dump(module, module_name)


def forward_acl_dump(module, module_name):
    global init_status
    if not init_status:
        init_status = True
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(DumpUtil.dump_config)
        torch_npu.npu.synchronize()
        module.forward(*module.input_args, **module.input_kwargs)
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
    del module.input_args
    del module.input_kwargs
    init_status = False
    print_info_log("Dump %s op file." % module_name)


def dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file):
    if "backward" in name_template:
        dump_tensor(out_feat, name_template.format("input"), dump_step, dump_file)
        dump_tensor(in_feat, name_template.format("output"), dump_step, dump_file)
    else:
        dump_tensor(in_feat, name_template.format("input"), dump_step, dump_file)
        dump_tensor(out_feat, name_template.format("output"), dump_step, dump_file)


def acc_cmp_dump(name, **kwargs):
    dump_step = kwargs.get('dump_step', 1)
    pid = kwargs.get('pid')
    DumpUtil.dump_config = kwargs.get('dump_config')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(module, in_feat, out_feat):
        if pid == os.getpid():
            dump_acc_cmp(name, in_feat, out_feat, dump_step, module)
        if hasattr(module, "input_args"):
            del module.input_args
        if hasattr(module, "input_kwargs"):
            del module.input_kwargs

    return acc_cmp_hook


def dump_overflow(module_name, stack_str, in_feat, out_feat, dump_file):
    name_template = f"{module_name}" + "_{}"
    DumpUtil.dump_data_dir = make_dump_data_dir(dump_file)
    _dump_tensor_completely(stack_str, name_template.format("stack_info"), dump_file)
    _dump_tensor_completely(in_feat, name_template.format("input"), dump_file)
    _dump_tensor_completely(out_feat, name_template.format("output"), dump_file)


def overflow_check(name, **kwargs):
    if DumpUtil.dump_path:
        DumpUtil.dump_dir = os.path.dirname(DumpUtil.dump_path)
    else:
        DumpUtil.dump_dir = './'
    overflow_nums = kwargs.get('overflow_nums', 1)
    pid = kwargs.get('pid')
    dump_mode = kwargs.get('dump_mode', "api")
    DumpUtil.dump_config = kwargs.get('dump_config')
    dump_config = kwargs.get('dump_config')
    if dump_mode == "acl":
        backward_obj = Backward()
        torch.autograd.backward = backward_obj.backward
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def overflowcheck_hook(module, in_feat, out_feat):
        if not OverFlowUtil.get_overflow_check_switch():
            return
        if pid != os.getpid():
            return
        if torch.cuda.is_available():
            print_warn_log("Overflow detection is not supported in the GPU environment.")
            return
        module_name = name
        module.has_overflow = torch_npu._C._check_overflow_npu()
        if not module.has_overflow:
            if hasattr(module, 'input_args'):
                del module.input_args
            if hasattr(module, 'input_kwargs'):
                del module.input_kwargs
        if module.has_overflow and OverFlowUtil.check_overflow_dump_times(overflow_nums):
            OverFlowUtil.inc_overflow_dump_times()
            dump_file_name = os.path.join(DumpUtil.dump_dir,
                "Overflow_info_{}_{}.pkl".format(get_time(), OverFlowUtil.real_overflow_dump_times))
            stack_str = []
            for (_, path, line, func, code, _) in inspect.stack()[3:]:
                if code:
                    stack_line = [path, str(line), func, code[0].strip()]
                else:
                    stack_line = [path, str(line), func, code]
                stack_str.append(stack_line)
            dump_overflow(module_name, stack_str, in_feat, out_feat, dump_file_name)

            print_warn_log("[overflow {} times]: module name :'{}' is overflow and dump file is saved in '{}'."
                           .format(OverFlowUtil.real_overflow_dump_times, module_name, os.path.realpath(dump_file_name)))
            if dump_mode == "acl":
                acl_dump(module, module_name)
           

            # clear overflow flag for the next check
            torch_npu._C._clear_overflow_npu()
            if not OverFlowUtil.check_overflow_dump_times(overflow_nums):
                raise ValueError("[overflow {} times]: dump file is saved in '{}'."
                                .format(OverFlowUtil.real_overflow_dump_times, os.path.realpath(dump_file_name)))
                return

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

    return overflowcheck_hook
