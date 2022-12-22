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

import os
import functools

import torch
if not torch.cuda.is_available():
    import torch_npu

from . import wrap_tensor, wrap_torch, wrap_functional
from .module import HOOKModule


def initialize_hook(hook):
    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith("wrap_"):
            setattr(torch.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))

    wrap_torch.wrap_torch_ops_and_bind(hook)
    for attr_name in dir(wrap_torch.HOOKTorchOP):
        if attr_name.startswith("wrap_"):
            setattr(torch, attr_name[5:], getattr(wrap_torch.HOOKTorchOP, attr_name))

    wrap_functional.wrap_functional_ops_and_bind(hook)
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(torch.nn.functional, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))


def register_hook(model, hook, **kwargs):
    assert hasattr(model, "named_modules"), "Please register hooks to nn.Module."

    dump_mode = kwargs.get('dump_mode', 1)
    pid = os.getpid()
    hook = functools.partial(hook, dump_mode=dump_mode, pid=pid)

    # In NPU scene, clear the overflow flag before overflow detection
    if not torch.cuda.is_available():
        torch_npu._C._clear_overflow_npu()

    initialize_hook(hook)
    for _, module in model.named_modules():
        if not isinstance(module, HOOKModule):
            continue

        prefix = "Module_" + module.__class__.__name__ + "_"
        if hasattr(module, "prefix_op_name_"):
            prefix = module.prefix_op_name_

        module.register_forward_hook(hook(prefix + "forward"))
        module.register_backward_hook(hook(prefix + "backward"))
