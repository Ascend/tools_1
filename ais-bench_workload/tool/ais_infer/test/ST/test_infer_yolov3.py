#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

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
        return "yolov3"

    def get_model_base_path(self):
        """
        supported model names as yolo, resnet50, resnet101,...。folder struct as follows
        testdata
         └── yolo   # model base
            ├── input
            ├── model
            └── output
        """
        return os.path.join(TestCommonClass.base_path, self.model_name)

    def get_dynamic_batch_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_yolov3_dymbatch.om")

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
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8]
        base_output_path = os.path.join(self.model_base_path, "output")
        output_paths = []

        for i, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            output_dirname = "{}_{}".format("static_batch", i)
            tmp_output_path = os.path.join(base_output_path, output_dirname)
            if os.path.exists(tmp_output_path):
                shutil.rmtree(tmp_output_path)
            os.makedirs(tmp_output_path)
            cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {}".format(TestCommonClass.cmd_prefix, model_path,
                                                                TestCommonClass.default_device_id, input_path, base_output_path, output_dirname)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_bin_file_num = len(os.listdir(tmp_output_path)) - 1
            assert(output_bin_file_num == 3 * self.output_file_num)
            output_paths.append(tmp_output_path)

        # compare different batchsize inference bin files
        base_compare_path = output_paths[0]
        exclude_file_name = "sumary.json"
        for i, cur_output_path in enumerate(output_paths):
            if i == 0:
                continue
            cmd = "diff -x {} {}  {}".format(exclude_file_name,
                                             base_compare_path,
                                             cur_output_path)
            ret = os.system(cmd)
            assert ret == 0

        for output_path in output_paths:
            shutil.rmtree(output_path)

    def test_general_inference_normal_dynamic_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8]
        base_output_path = os.path.join(self.model_base_path, "output")
        output_paths = []

        model_path = self.get_dynamic_batch_om_path()

        for i, batch_size in enumerate(batch_list):
            output_dirname = "{}_{}".format("static_batch", i)
            tmp_output_path = os.path.join(base_output_path, output_dirname)
            if os.path.exists(tmp_output_path):
                shutil.rmtree(tmp_output_path)
            os.makedirs(tmp_output_path)
            cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {}".format(TestCommonClass.cmd_prefix, model_path,
                                                                TestCommonClass.default_device_id, input_path, base_output_path, output_dirname)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_bin_file_num = len(os.listdir(tmp_output_path)) - 1
            assert(output_bin_file_num == 3 * self.output_file_num)
            output_paths.append(tmp_output_path)

        # compare different batchsize inference bin files
        base_compare_path = output_paths[0]
        exclude_file_name = "sumary.json"
        for i, cur_output_path in enumerate(output_paths):
            if i == 0:
                continue
            cmd = "diff -x {} {}  {}".format(exclude_file_name,
                                             base_compare_path,
                                             cur_output_path)
            ret = os.system(cmd)
            assert ret == 0

        for output_path in output_paths:
            shutil.rmtree(output_path)
