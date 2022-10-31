#!/usr/bin/env python
# -*- coding: utf-8 -*-

import filecmp
import json
import math
import os

import aclruntime
import pytest
from test_common import TestCommonClass


class TestClass():
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        print('\n ---class level teardown_class')

    def init(self):
        self.model_name = self.get_model_name(self)
        self.model_base_path = self.get_model_base_path(self)
        self.output_file_num = 5

    def get_model_name(self):
        return "bert"

    def get_model_base_path(self):
        """
        supported model names as bert, resnet50, resnet101,...。folder struct as follows
        testdata
         └── bert   # model base
            ├── input
            ├── model
            └── output
        """
        return os.path.join(TestCommonClass.base_path, self.model_name)

    def get_dynamic_batch_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_bert_dymbatch.om")

    def test_pure_inference_normal_static_batch(self):
        """
        batch size 1,2,4,8
        """
        batch_list = [1, 2, 4, 8]

        for _, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            cmd = "{} --model {} --device {}".format(TestCommonClass.cmd_prefix, model_path,
                                                     TestCommonClass.default_device_id)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_batch(self):
        batch_list = [1, 2, 4, 8]
        model_path = self.get_dynamic_batch_om_path()
        for _, dys_batch_size in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymBatch {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                                                   dys_batch_size)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_general_inference_normal_static_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_dir_path = TestCommonClass.get_bert_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num, "zero")
        input_paths = []
        input_paths.append(os.path.join(input_dir_path, "input_ids"))
        input_paths.append(os.path.join(input_dir_path, "input_mask"))
        input_paths.append(os.path.join(input_dir_path, "segment_ids"))
        input_path = ','.join(input_paths)
        batch_list = [1, 2, 4, 8]

        for _, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            cmd = "{} --model {} --device {} --input {}".format(TestCommonClass.cmd_prefix, model_path,
                                                                TestCommonClass.default_device_id, input_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_general_inference_normal_dynamic_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_dir_path = TestCommonClass.get_bert_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num, "zero")
        input_paths = []
        input_paths.append(os.path.join(input_dir_path, "input_ids"))
        input_paths.append(os.path.join(input_dir_path, "input_mask"))
        input_paths.append(os.path.join(input_dir_path, "segment_ids"))
        input_path = ','.join(input_paths)
        batch_list = [1, 2, 4, 8]

        for _, dys_batch_size in enumerate(batch_list):
            model_path = self.get_dynamic_batch_om_path()
            cmd = "{} --model {} --device {} --dymBatch {} --input {}".format(TestCommonClass.cmd_prefix, model_path,
                                                                              TestCommonClass.default_device_id,
                                                                              dys_batch_size,
                                                                              input_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

if __name__ == '__main__':
    pytest.main(['test_infer_bert.py', '-vs'])
