#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class is used to generate GUP dump data of the TensorFlow model.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
import pexpect
import readline
import time
import os
import numpy as np
import tensorflow as tf
from common.dump_data import DumpData
from common import utils
from common.utils import AccuracyCompareException


DTYPE_MAP = {
    tf.float32: np.float32,
    tf.float64: np.float64,
    tf.int64: np.int64,
    tf.int32: np.int32,
    tf.int16: np.int16,
    tf.int8: np.int8,
    tf.uint8: np.uint8,
    tf.bool: np.bool_,
    tf.complex64: np.complex64
}


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
        self.input_shapes = self._parse_input_shape()
        self.input_path = self.args.input_path

    def _parse_input_shape(self):
        """self.args.input_shape should be format like: tensor_name1:dim1,dim2;tensor_name2:dim1,dim2"""
        input_shapes = {}
        if self.args.input_shape == '':
            return input_shapes
        tensor_list = self.args.input_shape.split(';')
        for tensor in tensor_list:
            tensor_shape_list = tensor.split(':')
            if len(tensor_shape_list) == 2:
                input_shapes[tensor_shape_list[0]] = tensor_shape_list[1].split(',')
        return input_shapes

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

    def _get_inputs_tensor(self):
        inputs_tensor = []
        tensor_index = {}
        operations = self.global_graph.get_operations()
        for op in operations:
            # the operator with the 'Placeholder' type is the input operator of the model
            if "Placeholder" == op.type:
                op_name = op.name
                if op_name in tensor_index:
                    tensor_index[op_name] += 1
                else:
                    tensor_index[op_name] = 0
                tensor = self.global_graph.get_tensor_by_name(op.name + ":" + str(tensor_index[op_name]))
                tensor = self._verify_and_adapt_dynamic_shape(op.name, tensor)
                inputs_tensor.append(tensor)
        utils.print_info_log("model inputs tensor:\n{}\n".format(inputs_tensor))
        return inputs_tensor

    def _verify_and_adapt_dynamic_shape(self, op_name, tensor):
        try:
            model_shape = list(tensor.shape)
        except ValueError:
            if op_name not in self.input_shapes:
                utils.print_error_log("can not get model input tensor shape, and not set input shape in arguments.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
            tensor.set_shape(self.input_shapes[op_name])
            return tensor
        if op_name in self.input_shapes:
            fixed_tensor_shape = self.input_shapes[op_name]
            if len(fixed_tensor_shape) != len(model_shape):
                utils.print_error_log("fixed input tensor shape not equal to model input shape."
                                      " tensor_name:%s, %s vs %s" % (op_name, str(fixed_tensor_shape),
                                                                     str(model_shape)))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
            for index, dim in enumerate(model_shape):
                fixed_tensor_dim = fixed_tensor_shape[index]
                if dim is not None and fixed_tensor_dim != dim:
                    utils.print_error_log("fixed input tensor dim not equal to model input dim."
                                          " tensor_name:%s, %s vs %s" % (op_name, str(fixed_tensor_shape),
                                                                         str(model_shape)))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
                model_shape[index] = fixed_tensor_dim
            utils.print_info_log("Fix dynamic input shape of %s to %s" % (op_name, model_shape))
        tensor.set_shape(model_shape)
        return tensor

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
            outputs_tensor = self.args.output_nodes.split(';')
            for tensor in outputs_tensor:
                output_name = tensor.split(':')[0]
                if output_name not in node_list:
                    utils.print_error_log("The output node (%d) not in the graph. Please check the output node." % output_name)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
        else:
            output_nodes = list(set(node_list).difference(set(input_nodes)))
            for name in output_nodes:
                outputs_tensor.append(name + ":0")
        utils.print_info_log("The outputs tensor:\n{}\n".format(outputs_tensor))
        return outputs_tensor

    def _get_inputs_data(self, inputs_tensor):
        inputs_map = {}
        if "" == self.args.input_path:
            input_path_list = []
            for index, tensor in enumerate(inputs_tensor):
                input_data = np.random.random(self._convert_tensor_shape(tensor.shape)).astype(
                    self._convert_to_numpy_type(tensor.dtype))
                inputs_map[tensor] = input_data
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
            for index, tensor in enumerate(inputs_tensor):
                try:
                    input_data = np.fromfile(input_path[index], self._convert_to_numpy_type(tensor.dtype)).reshape(
                        self._convert_tensor_shape(tensor.shape))
                    inputs_map[tensor] = input_data
                    utils.print_info_log("load file name: {}, shape: {}, dtype: {}".format(
                        os.path.basename(input_path[index]), input_data.shape, input_data.dtype))
                except Exception as err:
                    utils.print_error_log("Failed to load data %s. %s" % (input_path[index], err))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
        return inputs_map

    @staticmethod
    def _convert_tensor_shape(tensor_shape):
        try:
            tensor_shape_list = tensor_shape.as_list()
            for i in range(len(tensor_shape_list)):
                if tensor_shape_list[i] is None:
                    utils.print_error_log("dynamic shape {} are not supported".format(tensor_shape))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_NOT_SUPPORT_ERROR)
        except Exception:
            utils.print_error_log("can not get model input tensor shape, please make input data by yourself")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_TENSOR_TYPE_ERROR)
        return tuple(tensor_shape_list)

    @staticmethod
    def _convert_to_numpy_type(tensor_type):
        np_type = DTYPE_MAP.get(tensor_type)
        if np_type is not None:
            return np_type
        utils.print_error_log("unsupported tensor type: {},".format(tensor_type))
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_TENSOR_TYPE_ERROR)

    def _run_model(self, outputs_tensor):
        cmd = 'tf_dbg_dump_data.py -m %s -i %s -o %s --output-nodes %s' \
              % (self.args.model_path, self.input_path, self.tf_dump_data_dir, ";".join(outputs_tensor))
        if self.args.input_shape:
            cmd += " -s " + self.args.input_shape
        self._run_tf_dbg_dump(cmd)

    def _run_tf_dbg_dump(self, cmd_line):
        """Run tf debug with pexpect, should set tf debug ui_type='readline'"""
        tf_dbg = pexpect.spawn(cmd_line)
        tf_dbg.expect('tfdbg>')
        tf_dbg.sendline('run')
        tf_dbg.expect('tfdbg>')
        tensor_name_path = os.path.join(self.tmp_dir, 'tf_tensor_names.txt')
        tf_dbg.sendline('lt > %s' % tensor_name_path)
        utils.print_error_log("Generate tensor name file.")
        tf_dbg.expect('tfdbg>')
        if not os.path.exists(tensor_name_path):
            utils.print_error_log("Failed to get tensor name in tf_debug.")
            raise AccuracyCompareException("Get tensor name in tf_debug failed.")
        utils.print_info_log("Save tensor name success. Generate tf dump commands from file: %s", tensor_name_path)
        tensor_dump_cmd_path = os.path.join(self.tmp_dir, 'tf_tensor_cmd.txt')
        convert_cmd = "timestamp=" + str(int(time.time())) + "; cat " + tensor_name_path + \
                      " | awk '{print \"pt\",$4,$4}'| awk '{gsub(\"/\", \"_\", $3); gsub(\":\", \".\", $3);" \
                      "print($1,$2,\"-n 0 -w " + self.tf_dump_data_dir + "/" + \
                      "\"$3\".\"\"'$timestamp'\"\".npy\")}' > " + tensor_dump_cmd_path
        util.execute_command(convert_cmd)
        if not os.path.exists(tensor_dump_cmd_path):
            utils.print_error_log("Save tf dump cmd failed")
            raise AccuracyCompareException("Failed to generate tf dump command.")
        utils.print_info_log("Generate tf dump commands. Start run commands in file: %s", tensor_dump_cmd_path)
        for cmd in open(tensor_dump_cmd_path):
            tf_dbg.sendline(cmd.strip())
            tf_dbg.expect('tfdbg>')
        tf_dbg.sendline('exit')
        utils.print_info_log('Finish dump tf data.')

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
        self._create_dir()
        self._load_graph()
        inputs_tensor = self._get_inputs_tensor()
        inputs_map = self._get_inputs_data(inputs_tensor)
        outputs_tensor = self._get_outputs_tensor()
        self._run_model(outputs_tensor)
        return self.tf_dump_data_dir
