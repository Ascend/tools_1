#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves generate npu dump data function.
Copyright Information:
HuaWei Technologies Co.,Ltd. All Rights Reserved © 2021
"""
import json
import os
import re

import numpy as np

from common import utils
from common.dump_data import DumpData
from common.utils import AccuracyCompareException

MSAME_DIR = "msame"
BUILD_SH = "build.sh"
OUT_PATH = "out"
MSAME_COMMAND_PATH = "msame"
ACL_JSON_PATH = "out/acl.json"
NPU_DUMP_DATA_BASE_PATH = "dump_data/npu"
RESULT_DIR = "result"
INPUT = "input"
GRAPH_OBJECT = "graph"
OP_OBJECT = "op"
NAME_OBJECT = "name"
TYPE_OBJECT = "type"
INPUT_DESC_OBJECT = "input_desc"
ATTR_OBJECT = "attr"
SHAPE_OBJECT = "shape"
DIM_OBJECT = "dim"
DATA_OBJECT = "Data"
NET_OUTPUT_OBJECT = "NetOutput"
ATC_CMDLINE_OBJECT = "atc_cmdline"
INPUT_SHAPE_RANGE = "--input_shape_range"
KEY_OBJECT = "key"
VALUE_OBJECT = "value"
S_OBJECT = "s"
DTYPE_OBJECT = "dtype"
DTYPE_MAP = {"DT_FLOAT": np.float32, "DT_FLOAT16": np.float16, "DT_DOUBLE": np.float64, "DT_INT8": np.int8,
             "DT_INT16": np.int16, "DT_INT32": np.int32, "DT_INT64": np.int64, "DT_UINT8": np.uint8,
             "DT_UINT16": np.uint16, "DT_UINT32": np.uint32, "DT_UINT64": np.uint64, "DT_BOOL": np.bool}


class NpuDumpData(DumpData):
    """
    Class for generate npu dump data
    """

    def __init__(self, arguments, output_json_path):
        self.arguments = arguments
        self.output_json_path = output_json_path
        self.json_object = self.load_json_file(self.output_json_path)
        self.shape_range = self._is_input_shape_range()

    def generate_dump_data(self):
        """
        Function Description:
            compile and rum msame project
        Return Value:
            npu dump data path
        """
        self._check_input_path_param()
        msame_dir = os.path.join(os.path.realpath(".."), MSAME_DIR)
        self.msame_compile(msame_dir)
        return self.msame_run(msame_dir)

    def msame_compile(self, msame_dir):
        """
        Function Description:
            compile msame project
        Parameter:
            msame_dir: msame project directory
        """
        execute_path = os.path.join(msame_dir, OUT_PATH, MSAME_COMMAND_PATH)
        if os.path.exists(execute_path):
            utils.print_info_log("The execute file %s exist. Skip the compile step." % execute_path)
            return
        utils.print_info_log("Start to compile %s" % msame_dir)
        out_path = os.path.join(msame_dir, OUT_PATH)
        build_sh_cmd = ["sh", BUILD_SH, "g++", out_path]
        os.chdir(msame_dir)
        # do build.sh command
        utils.print_info_log("Run command line: cd %s && %s" % (msame_dir, " ".join(build_sh_cmd)))
        utils.execute_command(build_sh_cmd)
        utils.print_info_log("Finish to compile %s." % msame_dir)

    def _make_msame_cmd_for_shape_range(self, msame_cmd):
        pattern = re.compile(r'^[0-9]+$')
        if self.shape_range:
            if not self.arguments.input_shape:
                utils.print_error_log('In the dynamic shape scenario, the "-s" or "--input-shape" is mandatory.')
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            msame_cmd.append('--dymShape')
            msame_cmd.append(self.arguments.input_shape)
            count = self._get_net_output_count()
            if not self.arguments.output_size:
                if count > 0:
                    count_list = []
                    for _ in range(count):
                        count_list.append("90000000")
                    self.arguments.output_size = ",".join(count_list)
            if self.arguments.output_size:
                output_size_list = self.arguments.output_size.split(',')
                if output_size_list != count:
                    utils.print_error_log('In the dynamic shape scenario, the "-s" or "--input-shape" is mandatory.')
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
                for item in output_size_list:
                    match = pattern.match(item)
                    if match is None:
                        utils.print_error_log("The size (%s) is invalid. Please check the output size."
                                              % self.arguments.output_size)
                        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
                    if int(item) <= 0:
                        utils.print_error_log("The size (%s) must be large than zero. Please check the output size."
                                              % self.arguments.output_size)
                        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
                msame_cmd.append('--outputSize')
                msame_cmd.append(self.arguments.output_size)

    def msame_run(self, msame_dir):
        """
        Function Description:
            run msame project
        Parameter:
            msame_dir: msame project directory
        Return Value:
            npu dump data path
        Exception Description:
            when invalid npu dump data path throw exception
        """
        self._compare_shape_vs_bin_file()
        npu_data_output_dir = os.path.join(self.arguments.out_path, NPU_DUMP_DATA_BASE_PATH)
        utils.create_directory(npu_data_output_dir)
        model_name, extension = utils.get_model_name_and_extension(self.arguments.offline_model_path)
        acl_json_path = os.path.join(msame_dir, ACL_JSON_PATH)
        if not os.path.exists(acl_json_path):
            os.mknod(acl_json_path, mode=0o600)
        self._write_content_to_acl_json(acl_json_path, model_name, npu_data_output_dir)
        msame_cmd = ["./" + MSAME_COMMAND_PATH, "--model", self.arguments.offline_model_path, "--input",
                     self.arguments.input_path, "--output", npu_data_output_dir]
        self._make_msame_cmd_for_shape_range(msame_cmd)
        os.chdir(os.path.join(msame_dir, OUT_PATH))
        # do msame command
        utils.print_info_log("Run command line: cd %s && %s" % (os.path.join(msame_dir, OUT_PATH), " ".join(msame_cmd)))
        utils.execute_command(msame_cmd)
        npu_dump_data_path, file_is_exist = utils.get_dump_data_path(npu_data_output_dir)
        if not file_is_exist:
            utils.print_error_log("The path {} dump data is not exist.".format(npu_dump_data_path))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        return npu_dump_data_path

    def _check_input_path_param(self):
        if self.arguments.input_path == "":
            input_path = os.path.join(self.arguments.out_path, INPUT)
            utils.check_file_or_directory_path(os.path.realpath(input_path), True)
            input_bin_files = os.listdir(input_path)
            input_bin_files.sort(key=lambda file: int((re.findall("\\d+", file))[0]))
            bin_file_path_array = []
            for item in input_bin_files:
                bin_file_path_array.append(os.path.join(input_path, item))
            self.arguments.input_path = ",".join(bin_file_path_array)
        else:
            bin_file_path_array = utils.check_input_bin_file_path(self.arguments.input_path)
            self.arguments.input_path = ",".join(bin_file_path_array)

    def _compare_shape_vs_bin_file(self):
        shape_size_array = self._get_shape_size()
        bin_files_size_array = self._get_bin_file_size()
        self._shape_size_vs_bin_file_size(shape_size_array, bin_files_size_array)

    def _get_shape_size(self):
        op_array = self._get_op_by_type()
        input_desc_array = self._get_input_desc_list(op_array)
        # extracts the input shape value
        return self._process_inputs(input_desc_array)

    def load_json_file(self, json_file_path):
        """
        Function Description:
            load json file
        Parameter:
            json_file_path: json file path
        Return Value:
            json object
        Exception Description:
            when invalid json file path throw exception
        """
        try:
            with open(json_file_path, "r") as input_file:
                try:
                    return json.load(input_file)
                except Exception as load_input_file_except:
                    print(str(load_input_file_except))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)
        except IOError as input_file_open_except:
            utils.print_error_log('Failed to open"' + json_file_path + '", ' + str(input_file_open_except))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR)

    def _get_op_by_type(self):
        op_array = []
        for graph in self.json_object.get(GRAPH_OBJECT):
            for operator in graph.get(OP_OBJECT):
                if DATA_OBJECT == operator.get(TYPE_OBJECT):
                    op_array.append(operator)
        return op_array

    def _get_net_output_count(self):
        count = 0
        for graph in self.json_object.get(GRAPH_OBJECT):
            for operator in graph.get(OP_OBJECT):
                if NET_OUTPUT_OBJECT == operator.get(TYPE_OBJECT) and INPUT_DESC_OBJECT in operator:
                    count += len(operator.get(INPUT_DESC_OBJECT))
        return count

    def _is_input_shape_range(self):
        if ATTR_OBJECT not in self.json_object:
            return False
        for attr in self.json_object.get(ATTR_OBJECT):
            if KEY_OBJECT in attr and attr.get(KEY_OBJECT) == ATC_CMDLINE_OBJECT:
                if VALUE_OBJECT in attr and S_OBJECT in attr.get(VALUE_OBJECT):
                    if INPUT_SHAPE_RANGE in attr.get(VALUE_OBJECT).get(S_OBJECT):
                        return True
        return False

    @staticmethod
    def _get_input_desc_list(op_array):
        input_desc_list = []
        for operator in op_array:
            if len(operator.get(INPUT_DESC_OBJECT)) != 0:
                for item in operator.get(INPUT_DESC_OBJECT):
                    input_desc_list.append(item)
        return input_desc_list

    @staticmethod
    def _process_inputs(input_desc_array):
        value = []
        for input_object in input_desc_array:
            if SHAPE_OBJECT not in input_object:
                value.append(0)
                continue
            item_sum = 1
            for num in input_object.get(SHAPE_OBJECT).get(DIM_OBJECT):
                item_sum *= num
            data_type = DTYPE_MAP.get(input_object.get(DTYPE_OBJECT))
            if not data_type:
                utils.print_error_log(
                    "The dtype attribute does not support {} value.".format(input_object[DTYPE_OBJECT]))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_KEY_ERROR)
            value.append(item_sum * np.dtype(data_type).itemsize)
        return value

    def _get_bin_file_size(self):
        bin_file_size = []
        bin_files = self.arguments.input_path.split(",")
        for item in bin_files:
            bin_file_size.append(os.path.getsize(item))
        return bin_file_size

    def _shape_size_vs_bin_file_size(self, shape_size_array, bin_files_size_array):
        if len(shape_size_array) < len(bin_files_size_array):
            utils.print_error_log("The number of input bin files is incorrect.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
        if self.shape_range:
            for bin_file_size in bin_files_size_array:
                if bin_file_size not in shape_size_array:
                    utils.print_error_log(
                        "The size (%d) of bin file can not match the input of the model." % bin_file_size)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
        else:
            for shape_size, bin_file_size in zip(shape_size_array, bin_files_size_array):
                if shape_size == 0:
                    continue
                if shape_size != bin_file_size:
                    utils.print_error_log("The shape value is different from the size of the bin file.")
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)

    @staticmethod
    def _write_content_to_acl_json(acl_json_path, model_name, npu_data_output_dir):
        load_dict = {
            "dump": {"dump_list": [{"model_name": model_name}], "dump_path": npu_data_output_dir, "dump_mode": "all",
                     "dump_op_switch": "off"}}
        if os.access(acl_json_path, os.W_OK):
            try:
                with open(acl_json_path, "w") as write_json:
                    try:
                        json.dump(load_dict, write_json)
                    except ValueError as write_json_except:
                        print(str(write_json_except))
                        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_WRITE_JSON_FILE_ERROR)
            except IOError as acl_json_file_except:
                utils.print_error_log('Failed to open"' + acl_json_path + '", ' + str(acl_json_file_except))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR)
        else:
            utils.print_error_log(
                "The path {} does not have permission to write.Please check the path permission".format(acl_json_path))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)
