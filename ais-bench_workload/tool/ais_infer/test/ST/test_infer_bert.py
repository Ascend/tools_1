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
        return os.path.join(self.model_base_path, "model",
                            "pth_bert_dymbatch.om")


    def test_pure_inference_normal_static_batch(self):
        """
        batch size 1,2,4,8
        """
        batch_list = [1, 2, 4, 8, 16]

        for _, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(
                batch_size, self.model_name)
            cmd = "{} --model {} --device {}".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0


    def test_pure_inference_normal_dynamic_batch(self):
        batch_list = [1, 2, 4, 8, 16]
        model_path = self.get_dynamic_batch_om_path()
        for _, dys_batch_size in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymBatch {}".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, dys_batch_size)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_general_inference_normal_static_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_ids_dir_path = TestCommonClass.get_inputs_path(
            input_size, os.path.join(self.model_base_path, "input",
                                     "input_ids"), self.output_file_num, "zero")
        input_mask_dir_path = TestCommonClass.get_inputs_path(
            input_size,
            os.path.join(self.model_base_path, "input", "input_mask"),
            self.output_file_num, "zero")
        segment_ids_dir_path = TestCommonClass.get_inputs_path(
            input_size,
            os.path.join(self.model_base_path, "input", "segment_ids"),
            self.output_file_num, "zero")
        input_paths = []
        input_paths.append(input_ids_dir_path)
        input_paths.append(input_mask_dir_path)
        input_paths.append(segment_ids_dir_path)
        input_path = ','.join(input_paths)
        batch_list = [1, 2, 4, 8, 16]

        base_output_path = os.path.join(self.model_base_path, "output")
        output_paths = []

        for i, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(
                batch_size, self.model_name)
            output_dirname = "{}_{}".format("static_batch", i)
            tmp_output_path = os.path.join(base_output_path, output_dirname)
            if os.path.exists(tmp_output_path):
                shutil.rmtree(tmp_output_path)
            os.makedirs(tmp_output_path)
            cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {}".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, input_path,
                base_output_path, output_dirname)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
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
        static_model_path = TestCommonClass.get_model_static_om_path(
            batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_ids_dir_path = TestCommonClass.get_inputs_path(
            input_size, os.path.join(self.model_base_path, "input",
                                     "input_ids"), self.output_file_num, "zero")
        input_mask_dir_path = TestCommonClass.get_inputs_path(
            input_size,
            os.path.join(self.model_base_path, "input", "input_mask"),
            self.output_file_num, "zero")
        segment_ids_dir_path = TestCommonClass.get_inputs_path(
            input_size,
            os.path.join(self.model_base_path, "input", "segment_ids"),
            self.output_file_num, "zero")
        input_paths = []
        input_paths.append(input_ids_dir_path)
        input_paths.append(input_mask_dir_path)
        input_paths.append(segment_ids_dir_path)
        input_path = ','.join(input_paths)
        batch_list = [1, 2, 4, 8, 16]
        base_output_path = os.path.join(self.model_base_path, "output")
        model_path = self.get_dynamic_batch_om_path()
        output_paths = []

        for i, dys_batch_size in enumerate(batch_list):
            output_dirname = "{}_{}".format("static_batch", i)
            tmp_output_path = os.path.join(base_output_path, output_dirname)
            if os.path.exists(tmp_output_path):
                shutil.rmtree(tmp_output_path)
            os.makedirs(tmp_output_path)
            cmd = "{} --model {} --device {} --dymBatch {} --input {} --output {} --output_dirname {} ".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, dys_batch_size, input_path,
                base_output_path, output_dirname)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
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


if __name__ == '__main__':
    pytest.main(['test_infer_bert.py', '-vs'])
