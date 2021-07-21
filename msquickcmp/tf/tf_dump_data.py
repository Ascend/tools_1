#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class is used to generate GUP dump data of the TensorFlow model.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""

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
        self.input_shapes = self._parse_input_shape()

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
        data_dir = os.path.join(self.args.out_path, "input")
        utils.create_directory(data_dir)

        # create dump_data/tf directory
        tf_dump_data_dir = os.path.join(self.args.out_path, "dump_data/tf")
        utils.create_directory(tf_dump_data_dir)

        return data_dir, tf_dump_data_dir

    def _load_graph(self):
        with tf.io.gfile.GFile(self.args.model_path, 'rb') as f:
            global_graph_def = tf.compat.v1.GraphDef.FromString(f.read())

        global_graph = tf.Graph()
        with global_graph.as_default():
            tf.import_graph_def(global_graph_def, name='')

        utils.print_info_log("load model success")

        return global_graph

    def _get_inputs_tensor(self, global_graph):
        inputs_tensor = []
        tensor_index = {}
        operations = global_graph.get_operations()
        for op in operations:
            # the operator with the 'Placeholder' type is the input operator of the model
            if "Placeholder" == op.type:
                op_name = op.name
                if op_name in tensor_index:
                    tensor_index[op_name] += 1
                else:
                    tensor_index[op_name] = 0

                tensor = global_graph.get_tensor_by_name(op.name + ":" + str(tensor_index[op_name]))
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

    def _get_outputs_tensor(self, global_graph):
        outputs_tensor = []
        outputs_tensor_name = []
        operations = global_graph.get_operations()
        utils.print_info_log("model operations:\n{}\n".format(operations))

        for op in operations:
            outputs_tensor_temp = op.inputs
            for tensor in outputs_tensor_temp:
                outputs_tensor_name.append(tensor.name)

        for name in outputs_tensor_name:
            tensor = global_graph.get_tensor_by_name(name)
            outputs_tensor.append(tensor)

        utils.print_info_log("model outputs tensor name (including all nodes):\n{}\n".format(outputs_tensor_name))
        utils.print_info_log("model outputs tensor (including all nodes):\n{}\n".format(outputs_tensor))

        return outputs_tensor

    def _get_inputs_data(self, data_dir, inputs_tensor):
        inputs_map = {}
        if "" == self.args.input_path:
            for index, tensor in enumerate(inputs_tensor):
                input_data = np.random.random(self._convert_tensor_shape(tensor.shape)).astype(
                    self._convert_to_numpy_type(tensor.dtype))
                inputs_map[tensor] = input_data
                file_name = "input_" + str(index) + ".bin"
                input_data.tofile(os.path.join(data_dir, file_name))
                utils.print_info_log("file name: {}, shape: {}, dtype: {}".format(file_name, input_data.shape,
                                                                                  input_data.dtype))
        else:
            input_path = self.args.input_path.split(",")
            if len(inputs_tensor) != len(input_path):
                utils.print_error_log("the number of model inputs tensor is not equal the number of "
                                      "inputs data, inputs tensor is: {}, inputs data is: {}".format(
                    len(inputs_tensor), len(input_path)))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
            for index, tensor in enumerate(inputs_tensor):
                input_data = np.fromfile(input_path[index], self._convert_to_numpy_type(tensor.dtype)).reshape(
                    self._convert_tensor_shape(tensor.shape))
                inputs_map[tensor] = input_data
                utils.print_info_log("load file name: {}, shape: {}, dtype: {}".format(
                    os.path.basename(input_path[index]), input_data.shape, input_data.dtype))

        return inputs_map

    def _convert_tensor_shape(self, tensor_shape):
        try:
            tensor_shape_list = tensor_shape.as_list()
            for i in range(len(tensor_shape_list)):
                if tensor_shape_list[i] is None:
                    utils.print_error_log("dynamic shape {} are not supported".format(tensor_shape))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_NOT_SUPPORT_ERROR)
        except ValueError:
            utils.print_info_log("can not get model input tensor shape, please make input data by yourself")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_TENSOR_TYPE_ERROR)
        return tuple(tensor_shape_list)

    def _convert_to_numpy_type(self, tensor_type):
        np_type = DTYPE_MAP.get(tensor_type)
        if np_type is not None:
            return np_type
        else:
            utils.print_error_log("unsupported tensor type: {},".format(tensor_type))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_TENSOR_TYPE_ERROR)

    def _run_model(self, global_graph, inputs_map, outputs_tensor):
        config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.compat.v1.Session(graph=global_graph, config=config) as sess:
            return sess.run(outputs_tensor, feed_dict=inputs_map)

    def _save_dump_data(self, dump_bins, tf_dump_data_dir, outputs_tensor):
        tensor_index = {}
        res_idx = 0
        for tensor in outputs_tensor:
            tensor_name = tensor.name
            tensor_name = tensor_name.replace("/", "_")
            tensor_name = tensor_name[0:tensor_name.find(":")]

            if tensor_name in tensor_index:
                tensor_index[tensor_name] += 1
            else:
                tensor_index[tensor_name] = 0

            file_name = tensor_name + "." + str(tensor_index[tensor_name]) + "." + str(round(
                time.time() * 1000000)) + ".npy"
            np.save(os.path.join(tf_dump_data_dir, file_name), dump_bins[res_idx])
            res_idx += 1

        utils.print_info_log("dump data success")

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
        data_dir, tf_dump_data_dir = self._create_dir()
        global_graph = self._load_graph()
        inputs_tensor = self._get_inputs_tensor(global_graph)
        inputs_map = self._get_inputs_data(data_dir, inputs_tensor)
        outputs_tensor = self._get_outputs_tensor(global_graph)
        dump_bins = self._run_model(global_graph, inputs_map, outputs_tensor)
        self._save_dump_data(dump_bins, tf_dump_data_dir, outputs_tensor)
        return tf_dump_data_dir