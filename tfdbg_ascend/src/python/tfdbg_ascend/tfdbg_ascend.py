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
# ==============================================================================
"""

# Description: Common depends and micro defines for and only for data preprocess module
import sys
import os
import absl.logging as logging
import tensorflow as tf
import threading
import tfdbg_ascend
sys.path.append(os.path.dirname(tfdbg_ascend.__file__))
import _tfdbg_ascend as dbg_cfg

dump_config = dbg_cfg.cfg(1, "")
try:
    __handle = tf.load_op_library(os.path.dirname(tfdbg_ascend.__file__) + "/_tfdbg_ascend.so")
except Exception as e:
    logging.error(e)

_hacked_tensorflow_function = tf.function
_thread_local = threading.local()


def never_nested_function(func=None, *args, **kwargs):
    """never_nested_function."""
    def never_nested_decorator(f):
        """never_nested_decorator."""
        if kwargs.get('experimental_compile'):
            logging.info("Skip xla compile tf function %s in debug dump mode", f.__name__)
        kwargs['experimental_compile'] = False
        tf_decorated_func = _hacked_tensorflow_function(*args, **kwargs)(f)

        def wrapper(*func_args, **func_kwargs):
            """wrapper."""
            if not hasattr(_thread_local, "entrance_function"):
                _thread_local.entrance_function = None
            if _thread_local.entrance_function is not None:
                logging.info("Inlining nested tf function %s under %s in debug dump mode", f.__name__,
                             _thread_local.entrance_function)
                return f(*func_args, **func_kwargs)
            _thread_local.entrance_function = f.__name__
            result = tf_decorated_func(*func_args, **func_kwargs)
            _thread_local.entrance_function = None
            return result

        wrapper.__name__ = f.__name__  # We should never change origin function name in decorator
        return wrapper

    if func is not None:
        return never_nested_decorator(func)
    else:
        return never_nested_decorator


tf.function = never_nested_function


def enable():
    """enable."""
    dump_config.Enable()
    get_dump_switch()


def disable():
    """disable."""
    dump_config.Disable()


def get_dump_switch():
    """dump_switch."""
    if dump_config.GetDumpSwitch() != 0:
        print("The dump function is enabled.")
    else:
        print("The dump function is disabled.")


def set_dump_path(dump_path):
    dump_config.SetDumpPath(dump_path)
    print("The current dump Path is: %s" % (dump_config.GetDumpPath()))


def get_dump_path():
    curr_path = dump_config.GetDumpPath()
    if curr_path:
        print('The current dump Path is: %s' % curr_path)
    else:
        print('The dump path is not set. It is the default path.')
    return curr_path
