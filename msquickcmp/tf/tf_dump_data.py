#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class is used to generate GUP dump data of the TensorFlow model.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
import sys

import pexpect
import time
import os
import numpy as np
import tensorflow as tf
from common.dump_data import DumpData
from common import utils
from common.utils import AccuracyCompareException


class TfDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the TensorFlow model.
    """

    def __init__(self, arguments):
        self.args = arguments
        output_path = os.path.realpath(self.args.out_path)
        self.data_dir = os.path.join(output_path, "input")
        self.tf_dump_data_dir = os.path.join(output_path, "dump_data/tf")
        self.tmp_dir = os.path.join(output_path, "tmp")
        self.global_graph = None
        self.input_path = self.args.input_path

    def _create_dir(self):
        # create input directory
        utils.create_directory(self.data_dir)

        # create dump_data/tf directory
        utils.create_directory(self.tf_dump_data_dir)

        # create tmp directory
        utils.create_directory(self.tmp_dir)

    def _load_graph(self):
        try:
            with tf.io.gfile.GFile(self.args.model_path, 'rb') as f:
                global_graph_def = tf.compat.v1.GraphDef.FromString(f.read())
            self.global_graph = tf.Graph()
            with self.global_graph.as_default():
                tf.import_graph_def(global_graph_def, name='')
        except Exception as err:
            utils.print_error_log("Failed to load the model %s. %s" % (self.args.model_path, err))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR)
        utils.print_info_log("Load the model %s successfully." % self.args.model_path)

    def _make_inputs_data(self, inputs_tensor):
        if "" == self.args.input_path:
            input_path_list = []
            for index, tensor in enumerate(inputs_tensor):
                input_data = np.random.random(tensor.shape).astype(utils.convert_to_numpy_type(tensor.dtype))
                input_path = os.path.join(self.data_dir, "input_" + str(index) + ".bin")
                input_path_list.append(input_path)
                try:
                    input_data.tofile(input_path)
                except Exception as err:
                    utils.print_error_log("Failed to generate data %s. %s" % (input_path, err))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
                utils.print_info_log("file name: {}, shape: {}, dtype: {}".format(
                    input_path, input_data.shape, input_data.dtype))
                self.input_path = ','.join(input_path_list)
        else:
            input_path = self.args.input_path.split(",")
            if len(inputs_tensor) != len(input_path):
                utils.print_error_log("the number of model inputs tensor is not equal the number of "
                                      "inputs data, inputs tensor is: {}, inputs data is: {}".format(
                    len(inputs_tensor), len(input_path)))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

    def _run_model(self, outputs_tensor):
        tf_debug_runner_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../", "tf_debug_runner.py")
        cmd = '%s %s -m %s -i %s --output-nodes %s' \
              % (sys.executable, tf_debug_runner_path, self.args.model_path, self.input_path, ";".join(outputs_tensor))
        if self.args.input_shape:
            cmd += " -s " + self.args.input_shape
        self._run_tf_dbg_dump(cmd)

    def _run_tf_dbg_dump(self, cmd_line):
        """Run tf debug with pexpect, should set tf debug ui_type='readline'"""
        tf_dbg = pexpect.spawn(cmd_line)
        try:
            tf_dbg.expect('tfdbg>', timeout=utils.TF_DEBUG_TIMEOUT)
            tf_dbg.sendline('run')
            tf_dbg.expect('tfdbg>', timeout=utils.TF_DEBUG_TIMEOUT)
        except Exception as ex:
            utils.print_error_log("Failed to run command: %s. %s" % (cmd_line, ex))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PYTHON_COMMAND_ERROR)
        tensor_name_path = os.path.join(self.tmp_dir, 'tf_tensor_names.txt')
        tf_dbg.sendline('lt > %s' % tensor_name_path)
        utils.print_info_log("Generate tensor name file.")
        tf_dbg.expect('tfdbg>', timeout=utils.TF_DEBUG_TIMEOUT)
        if not os.path.exists(tensor_name_path):
            utils.print_error_log("Failed to get tensor name in tf_debug.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PYTHON_COMMAND_ERROR)
        utils.print_info_log("Save tensor name success. Generate tf dump commands from file: %s", tensor_name_path)
        tensor_dump_cmd_path = os.path.join(self.tmp_dir, 'tf_tensor_cmd.txt')
        convert_cmd = "timestamp=" + str(int(time.time())) + "; cat " + tensor_name_path + \
                      " | awk '{print \"pt\",$4,$4}'| awk '{gsub(\"/\", \"_\", $3); gsub(\":\", \".\", $3);" \
                      "print($1,$2,\"-n 0 -w " + self.tf_dump_data_dir + "/" + \
                      "\"$3\".\"\"'$timestamp'\"\".npy\")}' > " + tensor_dump_cmd_path
        utils.execute_command(convert_cmd)
        if not os.path.exists(tensor_dump_cmd_path):
            utils.print_error_log("Save tf dump cmd failed")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PYTHON_COMMAND_ERROR)
        utils.print_info_log("Generate tf dump commands. Start run commands in file: %s", tensor_dump_cmd_path)
        for cmd in open(tensor_dump_cmd_path):
            tf_dbg.sendline(cmd.strip())
            tf_dbg.expect('tfdbg>', utils=self.TF_DEBUG_TIMEOUT)
        tf_dbg.sendline('exit')
        utils.print_info_log('Finish dump tf data.')

    def _get_outputs_tensor(self):
        input_nodes = []
        node_list = []
        operations = self.global_graph.get_operations()
        for op in operations:
            node_list.append(op.name)
            for tensor in op.inputs:
                input_name = tensor.name.split(':')[0]
                if input_name not in input_nodes:
                    input_nodes.append(input_name)
        outputs_tensor = []
        if self.args.output_nodes:
            outputs_tensor = self.args.output_nodes.strip().split(';')
            for tensor in outputs_tensor:
                tensor_info = tensor.strip().split(':')
                if len(tensor_info) != 2:
                    utils.print_error_log(
                        'Invalid output nodes (%s). Only support "name1:0;name2:1". Please check the output node.' % tensor)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
                output_name = tensor_info[0].strip()
                if output_name not in node_list:
                    utils.print_error_log(
                        "The output node (%s) not in the graph. Please check the output node." % output_name)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
        else:
            output_nodes = list(set(node_list).difference(set(input_nodes)))
            for name in output_nodes:
                outputs_tensor.append(name + ":0")
        utils.print_info_log("The outputs tensor:\n{}\n".format(outputs_tensor))
        return outputs_tensor

    def generate_dump_data(self):
        """
        Function description:
            generate TensorFlow model dump data
        Parameter:
            none
        Return Value:
            TensorFlow model dump data directory
        Exception Description:
            none
        """
        self._load_graph()
        self._create_dir()
        inputs_tensor = utils.get_inputs_tensor(self.global_graph, self.args.input_shape)
        self._make_inputs_data(inputs_tensor)
        outputs_tensor = self._get_outputs_tensor()
        self._run_model(outputs_tensor)
        return self.tf_dump_data_dir
