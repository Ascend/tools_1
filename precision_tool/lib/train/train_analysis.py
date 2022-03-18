# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from ..adapter.tf_adapter import TfAdapter
from ..dump.tf_dump import TfDump
from ..util.util import util
from ..config import config as cfg
from ..util.precision_tool_exception import PrecisionToolException
from ..util.precision_tool_exception import catch_tool_exception


class TrainAnalysis(object):
    def __init__(self):
        self.log = util.get_log()
        self.tf_adapter = TfAdapter()

    @staticmethod
    def gen_feed_file_name(name):
        file_name = str(name).replace(':', '_').replace('/', '_') + '.npy'
        return os.path.join(cfg.TF_CKPT_INPUT_DIR, file_name)

    def _init_session(self, device='npu', action='dump'):
        """"""
        if device == 'npu':
            return tf.Session(self.tf_adapter.session_dump_config(None, action=action))
        sess = tf.Session(tf.ConfigProto())
        return self.tf_adapter.sess_dump(sess)

    @staticmethod
    def _load_train_graph(sess):
        if util.empty_dir(cfg.TF_CKPT_DIR):
            raise PrecisionToolException('checkpoint dir [%s] is empty, can not run train analysis process.')
        ckpt = tf.train.latest_checkpoint(cfg.TF_CKPT_DIR)
        meta_graph = tf.train.import_meta_graph(ckpt + '.meta')
        meta_graph.restore(sess, ckpt)
        return tf.get_default_graph()

    @staticmethod
    def _get_input_from_graph(graph):
        input_nodes = []
        tensor_index = {}
        for op in graph.get_operations():
            if 'Placeholder' == op.type:
                if op.name in tensor_index:
                    tensor_index[op.name] += 1
                else:
                    tensor_index[op.name] = 0
                node = graph.get_tensor_by_name(op.name + ':' + str(tensor_index[op.name]))
                input_nodes.append(node)
        return input_nodes

    def _get_input_tensors(self, input_nodes):
        feed_map = {}
        for node in input_nodes:
            file_name = self.gen_feed_file_name(node.name)
            if os.path.isfile(file_name):
                feed_map[node] = np.load(file_name)
            else:
                # TD data type
                feed_map[node] = np.random.random(node.shape)
        return feed_map

    def _build_feed_map(self, graph):
        input_nodes = self._get_input_from_graph(graph)
        return self._get_input_tensors(input_nodes)

    def _analysis(self, device, action='dump'):
        sess = self._init_session(device, action=action)
        graph = self._load_train_graph(sess)
        train_op = tf.get_collection(tf.GraphKeys.TRAIN_OP)
        feed_map = self._build_feed_map(graph)
        sess.run(train_op, feed_dict=feed_map)
        if device == 'cpu':
            tf_dump = TfDump()
            tf_dump.run_tf_dbg_dump()

    def run(self, device='all', action='dump'):
        """
        :param device: all | npu | cpu
        :param action: dump | overflow | fusion_switch | fusion_off
        :return:
        """
        if device == 'all':
            self._analysis('cpu', action)
            self._analysis('npu', action)
        else:
            self._analysis(device, action)
