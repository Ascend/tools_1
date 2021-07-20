#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class is used to generate GUP dump data of the TensorFlow model.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
import argparse
import sys
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from common.dump_data import DumpData
from common import utils
from common.utils import AccuracyCompareException


class TfDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the TensorFlow model.
    """

    def __init__(self, arguments):
        self.args = arguments
        self.global_graph = None
        self.input_shapes = utils.parse_input_shape(self.args.input_shape)
        self.input_path = self.args.input_path

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

    def _get_outputs_tensor(self):
        outputs_tensor = []
        for tensor_name in self.args.output_nodes.split(';'):
            tensor = self.global_graph.get_tensor_by_name(tensor_name)
            outputs_tensor.append(tensor)
        return outputs_tensor

    def _get_inputs_data(self, inputs_tensor):
        inputs_map = {}
        input_path = self.args.input_path.split(",")
        for index, tensor in enumerate(inputs_tensor):
            try:
                input_data = np.fromfile(input_path[index],
                                         utils.convert_to_numpy_type(tensor.dtype)).reshape(tensor.shape)
                inputs_map[tensor] = input_data
                utils.print_info_log("load file name: {}, shape: {}, dtype: {}".format(
                    os.path.basename(input_path[index]), input_data.shape, input_data.dtype))
            except Exception as err:
                utils.print_error_log("Failed to load data %s. %s" % (input_path[index], err))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
        return inputs_map

    def _run_model(self, inputs_map, outputs_tensor):
        config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.compat.v1.Session(graph=self.global_graph, config=config) as sess:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
            return sess.run(outputs_tensor, feed_dict=inputs_map)

    def generate_dump_data(self):
        """
        Function description:
            generate TensorFlow model dump data
        Parameter:
            none
        Exception Description:
            none
        """
        self._load_graph()
        inputs_tensor = utils.get_inputs_tensor(self.global_graph, self.args.input_shape)
        inputs_map = self._get_inputs_data(inputs_tensor)
        outputs_tensor = self._get_outputs_tensor()
        self._run_model(inputs_map, outputs_tensor)


def _make_dump_data_parser(parser):
    parser.add_argument("-m", "--model-path", dest="model_path", default="",
                        help="<Required> model_path,original model file path,for example,.pb", required=True)
    parser.add_argument("-i", "--input", dest="input", default="",
                        help="<Optional> Input data path of the model. Separate multiple inputs with semicolons(;)."
                             " E.g: 'input_name1:0=input_0.bin;input_name1:1=input_0.bin'", required=True)
    parser.add_argument("-o", "--out-path", dest="out_path", default="", help="<Optional> output result path",
                        required=True)
    parser.add_argument("-s", "--input-shape", dest="input_shape", default="",
                        help="<Optional> Shape of input shape. Separate multiple nodes with semicolons(;)."
                             " E.g: input_name1:1,224,224,3;input_name2:3,300")
    parser.add_argument("--output-nodes", dest="output_nodes", default="",
                        help="<Optional> Output nodes designated by user. Separate multiple nodes with semicolons(;)."
                             " E.g: node_name1:0;node_name2:1;node_name3:0", required=True)


def main():
    """
   Function Description:
       main process function
   Exception Description:
       exit the program when an AccuracyCompare Exception  occurs
   """
    parser = argparse.ArgumentParser()
    _make_dump_data_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    TfDumpData(args).generate_dump_data()


if __name__ == '__main__':
    main()
