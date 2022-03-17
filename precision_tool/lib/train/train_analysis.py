# coding=utf-8
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from ..util.util import util
from ..config import config as cfg


class TrainAnalysis(object):
    def __init__(self):
        self.log = util.get_log()
        self.cpu_sess = self._init_cpu_session()
        self.npu_sess = self._init_npu_session()

    @staticmethod
    def _init_cpu_session():
        sess = tf.Session(tf.ConfigProto())
        return tf_debug.DumpingDebugWrapperSession(sess, cfg.TF_DEBUG_DUMP_DIR)

    @staticmethod
    def _init_npu_session():
        """"""

    def _load_inputs(self):
        """"""


    def _load_ckpt(self, sess):
        ckpt = tf.train.latest_checkpoint(cfg.TF_CKPT_DIR)
        meta_graph = tf.train.import_meta_graph(ckpt + '.meta')
        meta_graph.restore(sess, ckpt)

    def run(self, action):
        """
        :param action: support 'dump | overflow'
        :return:
        """
