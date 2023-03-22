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
import multiprocessing
import os.path
import sys

import numpy as np
import pandas as pd

from ..common.utils import check_file_or_directory_path, add_time_as_suffix, \
    print_error_log, CompareException, Const, format_value


def correct_data(result):
    if result == Const.NAN:
        return result
    if float(result) > 0.99999:
        return '1.0'
    return result


def cosine_similarity(n_value, b_value):
    np.seterr(divide='ignore', invalid='ignore')
    if len(n_value) == 1:
        return format_value(1.0), "This tensor is scalar."
    num = n_value.dot(b_value)
    a_norm = np.linalg.norm(n_value)
    b_norm = np.linalg.norm(b_value)
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
            result = format_value(cos)
    result = correct_data(result)
    return result, message


def get_rmse(n_value, b_value):
    rmse = np.linalg.norm(n_value - b_value) / np.sqrt(len(n_value))
    if np.isnan(rmse):
        rmse = Const.NAN
    return rmse, ""


def get_mape(n_value, b_value):
    mape_val = np.sum(np.abs((n_value - b_value) / b_value)) / len(b_value) * 100
    mape = Const.NAN if np.isnan(mape_val) else str(round(mape_val, 4)) + '%'
    return mape, ""


def get_max_abs_err(n_value, b_value):
    temp_res = n_value - b_value
    max_value = np.max(np.abs(temp_res))
    return format_value(max_value), ""


def get_max_relative_err(n_value, b_value):
    np.seterr(divide='ignore', invalid='ignore')
    relative_err = np.divide((n_value - b_value), b_value)
    max_relative_err = np.max(np.abs(relative_err))
    if np.isnan(max_relative_err):
        message = 'cannot compare by MaxRelativeError, The data contains 0 or nan in dump data.'
        return Const.NAN, message
    return format_value(max_relative_err), ""


def check_op(npu_dict, bench_dict, shape_flag):
    a_op_name = [_.split('_', 1)[1] for _ in npu_dict["op_name"]]
    b_op_name = [_.split('_', 1)[1] for _ in bench_dict["op_name"]]
    if shape_flag:
        return a_op_name == b_op_name and npu_dict["input_struct"] == bench_dict["input_struct"] \
            and npu_dict["output_struct"] == bench_dict["output_struct"]
    else:
        return a_op_name == b_op_name


def merge_tensor(tensor_list):
    op_dict = {}
    op_dict["op_name"] = []
    op_dict["input_struct"] = []
    op_dict["output_struct"] = []
    op_dict["summery"] = []
    op_dict["stack_info"] = []

    for tensor in tensor_list:
        if tensor[0].find("stack_info") != -1:
            op_dict["stack_info"].append(tensor[1])
            break
        op_dict["op_name"].append(tensor[0])
        if tensor[0].find("input") != -1:
            op_dict["input_struct"].append((tensor[3], tensor[4]))
        elif tensor[0].find("output") != -1:
            op_dict["output_struct"].append((tensor[3], tensor[4]))

        if tensor[1] <= Const.DUMP_RATIO_MAX:
            op_dict["summery"].append(tensor[5])

    return op_dict


def read_op(ops_queue, pkl_file_handle, stack_mode):
    tensor_list = []
    read_err = False
    read_output_flag = {"last_line": False, "curr_line": False}
    end_flag = "stack_info" if stack_mode is True else "output"

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
            read_output_flag["curr_line"] = True if tensor_data[0].find(end_flag) != -1 else False

        if (read_output_flag.get("last_line") and not read_output_flag.get("curr_line")) \
                or (len(tensor_line) == 0 and read_output_flag.get("curr_line")):  # end of file scenario
            ops_queue.append(merge_tensor(tensor_list))
            # the pos of the handle needs to restore to the start of the next api.
            pkl_file_handle.seek(curr_pos, 0)
            break
        tensor_list.append(tensor_data)

    return not read_err


def match_op(npu_queue, bench_queue, shape_flag):
    if check_op(npu_queue[-1], bench_queue[-1], shape_flag):
        return len(npu_queue) - 1, len(bench_queue) - 1
    for b_index, b_op in enumerate(bench_queue[0: -1]):
        if check_op(npu_queue[-1], b_op, shape_flag):
            return len(npu_queue) - 1, b_index
    for n_index, n_op in enumerate(npu_queue[0: -1]):
        if check_op(n_op, bench_queue[-1], shape_flag):
            return n_index, len(bench_queue) - 1
    return -1, -1


def get_accuracy(result, n_dict, b_dict, summery_flag):
    index_out = 0
    npu_stack_info = n_dict.get("stack_info", None)
    bench_stack_info = b_dict.get("stack_info", None)

    for index, n_name in enumerate(n_dict["op_name"]):
        b_name = b_dict["op_name"][index]
        if n_name.find("input") != -1:
            n_struct = n_dict["input_struct"][index]
            b_struct = b_dict["input_struct"][index]
        else:
            n_struct = n_dict["output_struct"][index_out]
            b_struct = b_dict["output_struct"][index_out]
            index_out += 1
        err_msg = ""

        result_item = [n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1], " ", " "]
        if summery_flag[0]:
            summery_data = n_dict.get("summery")[index]
            result_item.extend(summery_data)
        if summery_flag[1]:
            summery_data = b_dict.get("summery")[index]
            result_item.extend(summery_data)
        result_item.append(err_msg)
        if npu_stack_info and bench_stack_info and index == 0:
            result_item.extend(npu_stack_info)

        result.append(result_item)


def _do_multi_process(input_parma, result_path):
    try:
        _handle_multi_process(compare_ops, input_parma, result_path, multiprocessing.Manager().RLock())
    except FileNotFoundError as error:
        print("File not Found. compare failed!")
        return
    except IOError as error:
        print("IOEError. compare failed!")
        return


def read_dump_path(result_path):
    try:
        csv_pd = pd.read_csv(result_path)
        npu_dump_name_list = csv_pd.iloc[0:, 0].tolist()
        bench_dump_name_list = csv_pd.iloc[0:, 1].tolist()
        op_name_mapping_dict = {}
        for index, _ in enumerate(npu_dump_name_list):
            npu_dump_name = npu_dump_name_list[index]
            bench_dump_name = bench_dump_name_list[index]
            op_name_mapping_dict[npu_dump_name] = [npu_dump_name, bench_dump_name]
        return op_name_mapping_dict
    except FileNotFoundError as error:
        print(error)
        raise FileNotFoundError(error)
    except IOError as error:
        print(error)
        raise IOError(error)


def _handle_multi_process(func, input_parma, result_path, lock):
    process_num = int((multiprocessing.cpu_count() + 1) / 2)
    op_name_mapping_dict = read_dump_path(result_path)
    op_names = []
    for _ in range(process_num):
        op_names.append([])
    all_op_names = list(op_name_mapping_dict.keys())
    for i, op_name in enumerate(all_op_names):
        op_names[i % process_num].append(op_name)
    all_tasks = []
    pool = multiprocessing.Pool(process_num)

    def err_call(args):
        try:
            pool.terminate()
            if os.path.exists(result_path):
                os.remove(result_path)
            sys.exit(args)
        except SystemExit as error:
            print('multiprocess compare failed! season:{}'.format(args))

    for process_idx, fusion_op_names in enumerate(op_names):
        idx = [process_num, process_idx]
        task = pool.apply_async(func,
                                args=(idx, fusion_op_names, op_name_mapping_dict, result_path, lock, input_parma),
                                error_callback=err_call)
        all_tasks.append(task)
    pool.close()
    pool.join()


def compare_ops(idx, fusion_op_names, dump_path_dict, result_path, lock, input_parma):
    cos_result = []
    max_err_result = []
    err_mess = []
    is_print_compare_log = input_parma.get("is_print_compare_log")
    for i, op_name in enumerate(fusion_op_names):
        if is_print_compare_log:
            print("start comapre: {}".format(op_name))
        cos_sim, max_abs_err, err_msg = compare_by_op(op_name, dump_path_dict, input_parma)
        if is_print_compare_log:
            print("[{}] Compare result: cosine {}, max_abs_err {}, {}".format(op_name, cos_sim, max_abs_err, err_msg))
        cos_result.append(cos_sim)
        max_err_result.append(max_abs_err)
        err_mess.append(err_msg)
    _save_cmp_result(idx, cos_result, max_err_result, err_mess, result_path, lock)


def _save_cmp_result(idx, cos_result, max_err_result, err_msg, result_path, lock):
    lock.acquire()
    try:
        csv_pd = pd.read_csv(result_path, dtype=str)
        process_num = idx[0]
        process_idx = idx[1]
        for i, _ in enumerate(cos_result):
            process_index = i * process_num + process_idx
            csv_pd.loc[process_index, "Cosine"] = cos_result[i]
            csv_pd.loc[process_index, "MaxAbsErr"] = max_err_result[i]
            csv_pd.loc[process_index, "Err_message"] = err_msg[i]
        csv_pd.to_csv(result_path, index=False)
    except FileNotFoundError as error:
        print(error)
        raise FileNotFoundError(error)
    except IOError as error:
        print(error)
        raise IOError(error)
    finally:
        lock.release()


def compare_by_op(op_name, op_name_mapping_dict, input_parma):
    npu_bench_name_list = op_name_mapping_dict[op_name]
    try:
        n_value = np.load(os.path.join(input_parma.get("npu_dump_data_dir"), npu_bench_name_list[0] + ".npy"))
        b_value = np.load(os.path.join(input_parma.get("bench_dump_data_dir"), npu_bench_name_list[1] + ".npy"))
        n_value = n_value.reshape(-1).astype(float)
        b_value = b_value.reshape(-1).astype(float)
    except IOError as error:
        return " ", "", "Dump file:{} not found".format(error.filename)
    err_msg = ""
    cos_sim, message = cosine_similarity(n_value, b_value)
    err_msg += message
    max_abs_err, _ = get_max_abs_err(n_value, b_value)
    return cos_sim, max_abs_err, err_msg


def check_file_mode(npu_pkl, bench_pkl, stack_mode):
    npu_pkl_name = os.path.split(npu_pkl)[-1]
    bench_pkl_name = os.path.split(bench_pkl)[-1]
    if stack_mode:
        if not (npu_pkl_name.startswith("api_stack") and bench_pkl_name.startswith("api_stack")):
            raise Exception("The current file does not contain stack information, please turn off the stack_mode")
    else:
        if npu_pkl_name.startswith("api_stack") or bench_pkl_name.startswith("api_stack"):
            raise Exception("The current file contains stack information, please turn on the stack_mode")

def compare_distributed(npu_dump_dir, bench_dump_dir):
    def extract_pkl_and_data_dir(dirname):
        pids = os.listdir(dirname)
        if not len(pids) == 1:
            msg = ("Multiple pids are detected in one rank"
            "This case is not supported by compare_distributed() because"
            "We do not know the matching of the pids."
            "You may manually match the pids and use copmare() to compare them")
            raise NotImplementedError(msg)
        pid = pids[0] 
        dirname = os.path.join(dirname, pid)
        for fname in os.listdir(dirname):
            full_path = os.path.join(dirname, fname)
            if os.path.isdir(full_path):
                dump_data_dir = full_path 
            else:
                pkl_path = full_path 
        return pkl_path, dump_data_dir 

    npu_ranks = os.listdir(npu_dump_dir)
    bench_ranks = os.listdir(bench_dump_dir)
    npu_ranks.sort()
    bench_ranks.sort() 
    for nr, br in zip(npu_ranks, bench_ranks):
        n_dir = os.path.join(npu_dump_dir, nr)
        b_dir = os.path.join(bench_dump_dir, br)
        npu_pkl_path, npu_dump_data_dir = extract_pkl_and_data_dir(n_dir)
        bench_pkl_path, bench_dump_data_dir = extract_pkl_and_data_dir(b_dir)
        dump_result_param = {
            'npu_pkl_path': npu_pkl_path,
            'bench_pkl_path': bench_pkl_path,
            'npu_dump_data_dir': npu_dump_data_dir,
            'bench_dump_data_dir': bench_dump_data_dir,
            'is_print_compare_log':True
        }
        compare(dump_result_param, './output', True, suffix=f'_rank{nr}-{br}')

def compare(input_parma, output_path, shape_flag=True, stack_mode=False, suffix=''):
    try:
        check_file_or_directory_path(input_parma.get("npu_pkl_path"), False)
        check_file_or_directory_path(input_parma.get("bench_pkl_path"), False)
        check_file_or_directory_path(input_parma.get("npu_dump_data_dir"), True)
        check_file_or_directory_path(input_parma.get("bench_dump_data_dir"), True)
        check_file_or_directory_path(output_path, True)
        npu_pkl = open(input_parma.get("npu_pkl_path"), "r")
        bench_pkl = open(input_parma.get("bench_pkl_path"), "r")
        check_file_mode(npu_pkl.name, bench_pkl.name, stack_mode)
        npu_summary = _get_summery_mode(npu_pkl, input_parma.get("npu_pkl_path"))
        bench_summary = _get_summery_mode(bench_pkl, input_parma.get("bench_pkl_path"))
        result = compare_process(npu_pkl, bench_pkl, [npu_summary, bench_summary], shape_flag, stack_mode)
        npu_pkl.close()
        bench_pkl.close()

        columns = ["NPU Name", "Bench Name", "NPU Tensor Dtype", "Bench Tensor Dtype",
                   "NPU Tensor Shape", "Bench Tensor Shape", "Cosine", "MaxAbsErr"]
        if npu_summary:
            columns.extend(["NPU max", "NPU min", "NPU mean"])
        if bench_summary:
            columns.extend(["Bench max", "Bench min", "Bench mean"])
        columns.extend(["Err_message"])
        if stack_mode:
            columns.extend(["NPU_Stack_Info"])
        result_df = pd.DataFrame(result, columns=columns)

        file_name = add_time_as_suffix("compare_result" + suffix)
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        result_df.to_csv(file_path, index=False)
    except CompareException as error:
        print_error_log('Compare failed, Please check it and do it again!')
        sys.exit(error.code)
    _do_multi_process(input_parma, file_path)


def parse(pkl_file, module_name_prefix):
    pkl_handle = open(pkl_file, "r")
    done = False
    title_printed = False
    while not done:
        pkl_line = pkl_handle.readline()
        if pkl_line == '\n':
            continue
        if len(pkl_line) == 0:
            done = True
            break

        msg = json.loads(pkl_line)
        info_prefix = msg[0]
        if not info_prefix.startswith(module_name_prefix):
            continue

        if info_prefix.find("stack_info") != -1:
            print("\nTrace back({}):".format(msg[0]))
            for item in reversed(msg[1]):
                print("  File \"{}\", line {}, in {}".format(item[0], item[1], item[2]))
                print("    {}".format(item[3]))
            continue
        if len(msg) > 5:
            summery_info = "  [{}][dtype: {}][shape: {}][max: {}][min: {}][mean: {}]" \
                .format(msg[0], msg[3], msg[4], msg[5][0], msg[5][1], msg[5][2])
            if not title_printed:
                print("\nStatistic Info:")
                title_printed = True
            print(summery_info)
    pkl_handle.close()


def compare_process(npu_pkl_handle, bench_pkl_handle, summary_flag, shape_flag, stack_mode):
    npu_ops_queue = []
    bench_ops_queue = []
    result = []
    while True:
        npu_file_flag = read_op(npu_ops_queue, npu_pkl_handle, stack_mode)
        bench_file_flag = read_op(bench_ops_queue, bench_pkl_handle, stack_mode)
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
    pkl_file_handle.seek(0, 0)
    return isinstance(tensor_data[1], int) and tensor_data[1] <= Const.DUMP_RATIO_MAX


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npu_pkl_path', type=str, required=True)
    parser.add_argument('--bench_pkl_path', type=str, required=True)
    parser.add_argument('--npu_dump_data_dir', type=str, required=True)
    parser.add_argument('--bench_dump_data_dir', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--shape', action='store_true', default=False,
                        help='Enforce tensor.shape is same when op matches')
    args = parser.parse_args()
    compare(args, args.out_path, args.shape)
