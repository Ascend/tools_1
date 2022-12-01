#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves tf common function.
Copyright Information:
HuaWei Technologies Co.,Ltd. All Rights Reserved © 2022
"""

__all__ = ["register_acc_cmp_hook", "set_dump_path", "seed_all", "compare"]


import os
import random
import torch
import numpy as np

from . import wrap_tensor, wrap_torch, wrap_functional
from .module import register_acc_cmp_hook
from .hooks import set_dump_path
from .acc_compare import compare


wrap_tensor.wrap_tensor_ops_and_bind()
for attr_name in dir(wrap_tensor.HOOKTensor):
    if attr_name.startswith("wrap_"):
        setattr(torch.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))


wrap_torch.wrap_torch_ops_and_bind()
for attr_name in dir(wrap_torch.HOOKTorchOP):
    if attr_name.startswith("wrap_"):
        setattr(torch, attr_name[5:], getattr(wrap_torch.HOOKTorchOP, attr_name))


wrap_functional.wrap_functional_ops_and_bind()
for attr_name in dir(wrap_functional.HOOKFunctionalOP):
    if attr_name.startswith("wrap_"):
        setattr(torch.nn.functional, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
