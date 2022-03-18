# coding=utf-8
import os
import tensorflow as tf
import numpy as np
from .lib.util.util import util
from .lib.train.train_analysis import TrainAnalysis
from .lib.config import config as cfg


class WrapperSession(tf.Session):
    def __init__(self, target='', graph=None, config=None):
        super().__init__(target, graph, config)
        self.log = util.get_log()
        self._create_dir()
        self.saver = tf.train.Saver()

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """ wrapper super.run() """
        self._before_run(feed_dict)
        super(tf.Session, self).run(fetches, feed_dict, options, run_metadata)
        self._after_run()

    @staticmethod
    def _create_dir():
        util.create_dir(cfg.TF_CKPT_DIR)
        util.create_dir(cfg.TF_CKPT_INPUT_DIR)

    def _save_data(self, feed, feed_val):
        self.log.info('Save: %s', feed)
        file_name = TrainAnalysis.gen_feed_file_name(feed.name)
        # str(feed.name).replace(':', '_').replace('/', '_') + '.npy'
        # file_name = os.path.join(cfg.TF_CKPT_INPUT_DIR, feed_name)
        np.save(file_name, feed_val)

    def _before_run(self, feed_dict):
        """
        save feed dict tensors
        :return: None
        """
        if feed_dict:
            self.log.info('Session run with feed_dict, will save feed dict.')
            for feed, feed_val in feed_dict.item():
                self._save_data(feed, feed_val)
        # Iterator case

    def _after_run(self):
        """
        save checkpoint for dump and
        :return:
        """
        self.saver.save(self, cfg.TF_CKPT_DIR)
