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


import argparse
import json
import os.path

import numpy as np
import pandas as pd

from ..common.utils import check_file_or_directory_path, add_time_as_suffix,\
    print_error_log, CompareException, Const, format_value


def cosine_similarity(a, b):
    np.seterr(divide='ignore', invalid='ignore')
    if len(a) == 1:
        return format_value(1.0), "This tensor is scalar."
    a, b = np.mat(a), np.mat(b)
    num = float(a * b.T)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    message = ''
    if a_norm <= Const.FLOAT_EPSILON and b_norm <= Const.FLOAT_EPSILON:
        result = '1.0'
    elif a_norm <= Const.FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity, All the data is Zero in npu dump data.'
        result = Const.NAN
    elif b_norm <= Const.FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data.'
        result = Const.NAN
    else:
        cos = num / (a_norm * b_norm)
        if np.isnan(cos):
            message = 'Cannot compare by Cosine Similarity, the dump data has NaN.'
            result = Const.NAN
        else:
            result = format_value(0.5 + 0.5 * cos)

    return result, message


def get_rmse(a, b):
    rmse = np.linalg.norm(a - b) / np.sqrt(len(a))
    if np.isnan(rmse):
        rmse = Const.NAN
    return rmse, ""


def get_mape(a, b):
    mape_val = sum(np.abs((a - b) / b)) / len(b) * 100
    mape = Const.NAN if np.isnan(mape_val) else str(round(mape_val, 4)) + '%'
    return mape, ""


def get_max_abs_err(a, b):
    max_value = 0.0
    for a_data, b_data in zip(a, b):
        temp_x = float(a_data)
        temp_y = float(b_data)
        abs_error = abs(temp_x - temp_y)
        if abs_error > max_value:
            max_value = abs_error
    return format_value(max_value), ""


def get_max_relative_err(a, b):
    np.seterr(divide='ignore', invalid='ignore')
    relative_err = np.divide((a - b), b)
    max_relative_err = np.max(np.abs(relative_err))
    if np.isnan(max_relative_err):
        message = 'cannot compare by MaxRelativeError, The data contains 0 or nan in dump data.'
        return Const.NAN, message
    return format_value(max_relative_err), ""


def check_op(a, b, shape_flag):
    a_op_name = [_.split('_', 1)[1] for _ in a["op_name"]]
    b_op_name = [_.split('_', 1)[1] for _ in b["op_name"]]
    if shape_flag:
        return a_op_name == b_op_name and a["input_struct"] == b["input_struct"] \
            and a["output_struct"] == b["output_struct"]
    else:
        return a_op_name == b_op_name


def merge_tensor(tensor_list):
    op_dict = {}
    op_dict["op_name"] = []
    op_dict["input_struct"] = []
    op_dict["output_struct"] = []
    op_dict["input_value"] = []
    op_dict["output_value"] = []
    op_dict["summery"] = []

    for tensor in tensor_list:
        if tensor[0].find("stack_info") != -1:
            continue
        op_dict["op_name"].append(tensor[0])
        if tensor[0].find("input") != -1:
            op_dict["input_struct"].append((tensor[3], tensor[4]))
            op_dict["input_value"].append(tensor[2])
        elif tensor[0].find("output") != -1:
            op_dict["output_struct"].append((tensor[3], tensor[4]))
            op_dict["output_value"].append(tensor[2])

        if tensor[1] == Const.DUMP_MODE.get("SUMMERY"):
            op_dict["summery"].append(tensor[5])

    return op_dict


def read_op(ops_queue, pkl_file_handle):
    tensor_list = []
    read_err = False
    read_output_flag = {"last_line": False, "curr_line": False}
    while True:
        curr_pos = pkl_file_handle.tell()
        tensor_line = pkl_file_handle.readline()
        if len(tensor_line) == 0 and not read_output_flag.get("curr_line"):
            read_err = True
            break
        if tensor_line == '\n':
            continue
        if len(tensor_line) != 0:
            tensor_data = json.loads(tensor_line)
            read_output_flag["last_line"] = read_output_flag.get("curr_line")
            read_output_flag["curr_line"] = True if tensor_data[0].find("output") != -1 else False

        if (read_output_flag.get("last_line") and not read_output_flag.get("curr_line")) or\
                (len(tensor_line) == 0 and read_output_flag.get("curr_line")):  # end of file scenario
            ops_queue.append(merge_tensor(tensor_list))
            # the pos of the handle needs to restore to the start of the next api.
            pkl_file_handle.seek(curr_pos, 0)
            break
        tensor_list.append(tensor_data)

    return not read_err


def match_op(npu_queue, bench_queue, shape_flag):
    if check_op(npu_queue[-1], bench_queue[-1], shape_flag):
        return len(npu_queue)-1, len(bench_queue)-1
    for b_index, b_op in enumerate(bench_queue[0: -1]):
        if check_op(npu_queue[-1], b_op, shape_flag):
            return len(npu_queue)-1, b_index
    for n_index, n_op in enumerate(npu_queue[0: -1]):
        if check_op(n_op, bench_queue[-1], shape_flag):
            return n_index, len(bench_queue)-1
    return -1, -1


def get_accuracy(result, n_dict, b_dict, summery_flag):
    for index, n_name in enumerate(n_dict["op_name"]):
        b_name = b_dict["op_name"][index]
        if n_name.find("input") != -1:
            n_value = np.array(n_dict["input_value"][index])
            b_value = np.array(b_dict["input_value"][index])
            n_struct = n_dict["input_struct"][index]
            b_struct = b_dict["input_struct"][index]
        else:
            n_value = np.array(n_dict["output_value"][0])
            b_value = np.array(b_dict["output_value"][0])
            n_struct = n_dict["output_struct"][0]
            b_struct = b_dict["output_struct"][0]
        err_msg = ""
        if n_struct[1] != b_struct[1]:
            cos_sim = "cannot be calculated "
            rmse = "cannot be calculated"
            mape = "cannot be calculated"
            max_abs_err = "cannot be calculated"
            max_relative_err = "cannot be calculated"
        else:
            cos_sim, message = cosine_similarity(n_value, b_value)
            err_msg += message
            rmse, _ = get_rmse(n_value, b_value)
            mape, _ = get_mape(n_value, b_value)
            max_abs_err, _ = get_max_abs_err(n_value, b_value)
            max_relative_err, message = get_max_relative_err(n_value, b_value)
            err_msg += message

        result_item = [n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1],
                       cos_sim, rmse, mape, max_abs_err, max_relative_err]
        if summery_flag[0]:
            summery_data = n_dict.get("summery")[index]
            result_item.extend(summery_data)
        if summery_flag[1]:
            summery_data = b_dict.get("summery")[index]
            result_item.extend(summery_data)
        result_item.append(err_msg)
        result.append(result_item)


def compare(npu_pkl_path, bench_pkl_path, output_path, shape_flag=False):
    check_file_or_directory_path(output_path, True)
    npu_pkl = open(npu_pkl_path, "r")
    bench_pkl = open(bench_pkl_path, "r")
    npu_summary = _get_summery_mode(npu_pkl, npu_pkl_path)
    bench_summary = _get_summery_mode(bench_pkl, bench_pkl_path)
    result = compare_process(npu_pkl, bench_pkl, [npu_summary, bench_summary], shape_flag)
    npu_pkl.close()
    bench_pkl.close()

    columns = ["NPU Name", "Bench Name", "NPU Tensor Dtype", "Bench Tensor Dtype",
               "NPU Tensor Shape", "Bench Tensor Shape", "Cosine", "RMSE", "MAPE", "MaxAbsErr", "MaxRelativeErr"]
    if npu_summary:
        columns.extend(["NPU max", "NPU min", "NPU mean"])
    if bench_summary:
        columns.extend(["Bench max", "Bench min", "Bench mean"])
    columns.extend(["Err_message"])
    result_df = pd.DataFrame(result, columns=columns)

    file_name = add_time_as_suffix("compare_result")
    file_path = os.path.join(os.path.realpath(output_path), file_name)
    result_df.to_csv(file_path, index=False)


def compare_process(npu_pkl_handle, bench_pkl_handle, summary_flag, shape_flag):
    npu_ops_queue = []
    bench_ops_queue = []
    result = []
    while True:
        npu_file_flag = read_op(npu_ops_queue, npu_pkl_handle)
        bench_file_flag = read_op(bench_ops_queue, bench_pkl_handle)
        if (not npu_file_flag and not bench_file_flag) \
                or (len(npu_ops_queue) == 0 or len(bench_ops_queue) == 0):
            break
        n_match_point, b_match_point = match_op(npu_ops_queue, bench_ops_queue, shape_flag)
        if n_match_point == -1 and b_match_point == -1:
            continue
        n_match_data = npu_ops_queue[n_match_point]
        b_match_data = bench_ops_queue[b_match_point]
        get_accuracy(result, n_match_data, b_match_data, summary_flag)
        del npu_ops_queue[0: n_match_point + 1]
        del bench_ops_queue[0: b_match_point + 1]
    return result


def _get_summery_mode(pkl_file_handle, file_name):
    tensor_line = pkl_file_handle.readline()
    if len(tensor_line) == 0:
        print_error_log("dump file {} have empty line!".format(file_name))
        raise CompareException(CompareException.INVALID_DUMP_FILE)
    tensor_data = json.loads(tensor_line)
    return tensor_data[1] == Const.DUMP_MODE.get("SUMMERY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npu_pkl', type=str, required=True)
    parser.add_argument('--bench_pkl', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--shape', action='store_true', default=False,
                    help='Enforce tensor.shape is same when op matches')
    args = parser.parse_args()
    compare(args.npu_pkl, args.bench_pkl, args.out_path, args.shape)
