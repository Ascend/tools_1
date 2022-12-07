#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import math
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
        for i, cur_output_path in enumerate(output_paths):
            if i == 0:
                continue
            cmd = "diff  {}  {}".format(base_compare_path, cur_output_path)
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
        for i, cur_output_path in enumerate(output_paths):
            if i == 0:
                continue
            cmd = "diff {}  {}".format(base_compare_path, cur_output_path)
            ret = os.system(cmd)
            assert ret == 0

        for output_path in output_paths:
            shutil.rmtree(output_path)

    def test_general_inference_prformance_comparison_with_msame_static_batch(self):
        batch_size = 1
        input_file_num = 100
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_ids_dir_path = TestCommonClass.get_inputs_path(
            input_size, os.path.join(self.model_base_path, "input",
                                     "input_ids"), input_file_num, "zero")
        input_mask_dir_path = TestCommonClass.get_inputs_path(
            input_size,
            os.path.join(self.model_base_path, "input", "input_mask"),
            input_file_num, "zero")
        segment_ids_dir_path = TestCommonClass.get_inputs_path(
            input_size,
            os.path.join(self.model_base_path, "input", "segment_ids"),
            input_file_num, "zero")
        input_paths = []
        input_paths.append(input_ids_dir_path)
        input_paths.append(input_mask_dir_path)
        input_paths.append(segment_ids_dir_path)
        input_path = ','.join(input_paths)

        output_path = os.path.join(self.model_base_path, "output")

        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        output_dir_name = "static_batch"
        output_dir_path = os.path.join(output_path, output_dir_name)
        summary_json_path = os.path.join(output_path, "{}_summary.json".format(output_dir_name))
        if os.path.exists(output_dir_path):
            shutil.rmtree(output_dir_path)
        os.makedirs(output_dir_path)
        cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {}".format(
            TestCommonClass.cmd_prefix, model_path,
            TestCommonClass.default_device_id, input_path,
            output_path, output_dir_name)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0

        with open(summary_json_path,'r',encoding='utf8') as fp:
            json_data = json.load(fp)
            ais_inference_time_ms = json_data["NPU_compute_time"]["mean"]
        assert math.fabs(ais_inference_time_ms) > TestCommonClass.EPSILON

        # get msame inference  average time without first time
        msame_infer_log_path = os.path.join(output_path, output_dir_name, "msame_infer.log")
        cmd = "{} --model {} --input {} > {}".format(TestCommonClass.msame_bin_path, model_path, input_path, msame_infer_log_path)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        assert os.path.exists(msame_infer_log_path)

        msame_inference_time_ms = 0
        with open(msame_infer_log_path) as f:
            for line in f:
                if "Inference average time without first time" not in line:
                    continue

                sub_str = line[(line.rfind(':') + 1):]
                sub_str = sub_str.replace('ms\n','')
                msame_inference_time_ms = float(sub_str)

        assert math.fabs(msame_inference_time_ms) > TestCommonClass.EPSILON
        # compare
        allowable_performance_deviation = 0.01
        assert math.fabs(msame_inference_time_ms - ais_inference_time_ms)/msame_inference_time_ms < allowable_performance_deviation
        os.remove(msame_infer_log_path)
        shutil.rmtree(output_dir_path)

    def test_general_inference_batchsize_is_none_normal_static_batch(self):
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

        output_parent_path = os.path.join(self.model_base_path, "output")
        output_paths = []
        summary_paths = []

        for i, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(
                batch_size, self.model_name)
            output_dirname = "{}_{}".format("static_batch", i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {} > {}".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, input_path,
                output_parent_path, output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert batch_size == cur_batchsize
                    break

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

    def test_general_inference_batchsize_is_none_normal_dynamic_batch(self):
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
        output_parent_path = os.path.join(self.model_base_path, "output")
        model_path = self.get_dynamic_batch_om_path()
        output_paths = []
        summary_paths = []

        for i, dys_batch_size in enumerate(batch_list):
            output_dirname = "{}_{}".format("dynamic_batch", i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            cmd = "{} --model {} --device {} --dymBatch {} --output {} --output_dirname {} > {}".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, dys_batch_size,
                output_parent_path, output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert dys_batch_size == cur_batchsize
                    break

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

if __name__ == '__main__':
    pytest.main(['test_infer_bert.py', '-vs'])
