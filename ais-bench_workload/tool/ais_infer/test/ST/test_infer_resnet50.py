import filecmp
import json
import math
import os
import shutil

import aclruntime
import numpy as np
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
        return "resnet50"

    def get_model_base_path(self):
        """
        supported model names as resnet50, resnet101,...。folder struct as follows
        testdata
         └── resnet50   # model base
            ├── input
            ├── model
            └── output
        """
        return os.path.join(TestCommonClass.base_path, self.model_name)

    def get_dynamic_batch_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymbatch.om")

    def get_dynamic_hw_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymwh.om")

    def get_dynamic_dim_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymdim.om")

    def get_dynamic_shape_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymshape.om")

    def create_npy_files_in_auto_set_dymshape_mode_input(self, dirname, shapes):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

        os.makedirs(dirname)

        i = 1
        for shape in shapes:
            x = np.zeros(shape, dtype=np.int32)
            file_name = 'input_shape_{}'.format(i)
            file = os.path.join(dirname, "{}.npy".format(file_name))
            np.save(file, x)
            i += 1

    def get_dynamic_shape_om_file_size(self, shape):
        """"
        dym_shape = "actual_input_1:1,3,224,224"
        """
        if len(shape) == 0:
            return 0

        sub_str = shape[(shape.rfind(':') + 1):]
        sub_str = sub_str.replace('\n','')
        num_arr = sub_str.split(',')
        fix_num = 4
        size = int(num_arr[0]) * int(num_arr[1]) * int(num_arr[2]) * int(num_arr[3]) * fix_num
        return size

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
        batch_list = [1, 2, 4, 8, 16]
        model_path = self.get_dynamic_batch_om_path()
        for _, dys_batch_size in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymBatch {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                                                   dys_batch_size)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_hw(self):
        batch_list = ["224,224", "448,448"]
        model_path = self.get_dynamic_hw_om_path()
        for _, dym_hw in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymHW {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id, dym_hw)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_dims(self):
        batch_list = ["actual_input_1:1,3,224,224", "actual_input_1:8,3,448,448"]

        model_path = self.get_dynamic_dim_om_path()
        for _, dym_dims in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymDims {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id, dym_dims)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_shape(self):
        dym_shape = "actual_input_1:1,3,224,224"
        output_size = 10000
        model_path = self.get_dynamic_shape_om_path()
        cmd = "{} --model {} --device {} --outputSize {} --dymShape {} ".format(TestCommonClass.cmd_prefix, model_path,
                                                                                TestCommonClass.default_device_id,
                                                                                output_size,
                                                                                dym_shape)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0

    def test_inference_normal_dynamic_shape_auto_set_dymshape_mode(self):
        """"
        multiple npy input files or a npy folder as input parameter
        """
        shapes = [[1, 3,  224,  224], [1, 3, 300, 300], [1, 3, 200, 200]]
        auto_set_dymshape_mode_input_dir_path = os.path.join(self.model_base_path, "input", "auto_set_dymshape_mode_input")
        self.create_npy_files_in_auto_set_dymshape_mode_input(auto_set_dymshape_mode_input_dir_path, shapes)

        output_size = 10000
        model_path = self.get_dynamic_shape_om_path()
        filelist = os.listdir(auto_set_dymshape_mode_input_dir_path)
        num_shape = len(filelist)
        file_paths = []
        for file in filelist:
            file_paths.append(os.path.join(auto_set_dymshape_mode_input_dir_path, file))
        file_paths = ",".join(file_paths)
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "auto_set_dymshape_mode_output"
        output_path = os.path.join(output_parent_path, output_dirname)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        cmd = "{} --model {} --device {} --outputSize {} --auto_set_dymshape_mode true --input {} --output {}  --output_dirname {} ".format(TestCommonClass.cmd_prefix, model_path,
            TestCommonClass.default_device_id, output_size, file_paths, output_parent_path, output_dirname)

        ret = os.system(cmd)
        assert ret == 0

        try:
            cmd = "find {} -name '*.bin'|wc -l".format(output_path)
            bin_num = os.popen(cmd).read()
        except Exception as e:
            raise Exception("raise an exception: {}".format(e))

        assert int(bin_num) == num_shape
        shutil.rmtree(output_path)
        os.makedirs(output_path)
        # check input parameter is a folder
        cmd = "{} --model {} --device {} --outputSize {} --auto_set_dymshape_mode true --input {} --output {}  --output_dirname {} ".format(TestCommonClass.cmd_prefix, model_path,
            TestCommonClass.default_device_id, output_size, auto_set_dymshape_mode_input_dir_path, output_parent_path, output_dirname)

        ret2 = os.system(cmd)
        assert ret2 == 0

        try:
            cmd = "find {} -name '*.bin'|wc -l".format(output_path)
            bin_num2 = os.popen(cmd).read()
        except Exception as e:
            raise Exception("raise an exception: {}".format(e))

        assert int(bin_num2) == num_shape
        shutil.rmtree(output_path)


    def test_general_inference_normal_static_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8, 16]

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
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8, 16]

        for _, dys_batch_size in enumerate(batch_list):
            model_path = self.get_dynamic_batch_om_path()
            cmd = "{} --model {} --device {} --dymBatch {} --input {}".format(TestCommonClass.cmd_prefix, model_path,
                                                                              TestCommonClass.default_device_id,
                                                                              dys_batch_size,
                                                                              input_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_general_inference_normal_run_mode(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        run_modes = ["array", "tensor"]
        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        for _, run_mode in  enumerate(run_modes):
            cmd = "{} --model {} --device {} --input {} --run_mode {}".format(TestCommonClass.cmd_prefix, model_path,
                                                                TestCommonClass.default_device_id, input_path, run_mode)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_general_inference_normal_inference_time(self):
        batch_size = 1
        num_input_file = 100
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)

        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     num_input_file)

        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "infer_time_output"
        output_path = os.path.join(output_parent_path, output_dirname)
        log_path = os.path.join(output_path, "log.txt")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        cmd = "{} --model {} --device {} --input {}  --debug true --output {}  --output_dirname {} > {}".format(TestCommonClass.cmd_prefix, model_path,
            TestCommonClass.default_device_id, input_path, output_parent_path, output_dirname, log_path)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        assert os.path.exists(log_path)

        # ignore of warmup inference time, get  inferening times from log file and  summary.json, compare them and assert
        infer_time_lists = []
        with open(log_path) as f:
            i = 0
            for line in f:
                if "cost :" not in line:
                    continue
                i += 1
                if i == 1:
                    continue

                sub_str = line[(line.rfind(':') + 1):]
                sub_str = sub_str.replace('\n','')
                infer_time_lists.append(float(sub_str))

        time_array = np.array(infer_time_lists)
        summary_json_path = os.path.join(output_parent_path, "{}_summary.json".format(output_dirname))
        with open(summary_json_path,'r',encoding='utf8') as fp:
            json_data = json.load(fp)
            json_mean = json_data["NPU_compute_time"]["mean"]
            json_percentile = json_data["NPU_compute_time"]["percentile(99%)"]

            assert math.fabs(time_array.mean() - json_mean) <= TestCommonClass.EPSILON
            assert math.fabs(np.percentile(time_array, 99) - json_percentile) <= TestCommonClass.EPSILON
        os.remove(summary_json_path)
        os.remove(log_path)
        shutil.rmtree(output_path)

    def test_general_inference_normal_warmup_count(self):
        batch_size = 1
        num_input_file = 10
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)

        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     num_input_file)

        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "warmup_output"
        output_path = os.path.join(output_parent_path, output_dirname)
        log_path = os.path.join(output_path, "log.txt")
        warmups = [-1, 0, 100]
        for i, warmup_num in enumerate(warmups):
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            cmd = "{} --model {} --device {} --input {}  --debug true --output {}  --output_dirname {} --warmup_count {} > {}".format(TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, input_path, output_parent_path, output_dirname,  warmup_num, log_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            if i == 0:
                assert ret != 0
            else:
                assert ret == 0
                assert os.path.exists(log_path)

                try:
                    cmd = "cat {} |grep 'cost :' | wc -l".format(log_path)
                    outval = os.popen(cmd).read()
                except Exception as e:
                    raise Exception("raise an exception: {}".format(e))

                assert int(outval) == (num_input_file + warmup_num)

                summary_json_path = os.path.join(output_parent_path, "{}_summary.json".format(output_dirname))
                with open(summary_json_path,'r',encoding='utf8') as fp:
                    json_data = json.load(fp)
                    NPU_compute_time_count = json_data["NPU_compute_time"]["count"]
                    h2d_num = json_data["H2D_latency"]["count"]
                    d2h_num = json_data["D2H_latency"]["count"]
                    assert NPU_compute_time_count == num_input_file
                    assert h2d_num == num_input_file
                    assert d2h_num == num_input_file
                os.remove(summary_json_path)
        shutil.rmtree(output_path)

    def test_pure_inference_normal_warmup_count_200(self):
        batch_size = 1
        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "warmup_output"
        output_path = os.path.join(output_parent_path, output_dirname)
        log_path = os.path.join(output_path, "log.txt")
        warmup_num = 200
        loop_num = 10

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        cmd = "{} --model {} --device {} --debug true --output {}  --output_dirname {} --warmup_count {} --loop {} > {}".format(TestCommonClass.cmd_prefix, model_path,
            TestCommonClass.default_device_id, output_parent_path, output_dirname,  warmup_num, loop_num, log_path)

        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        assert os.path.exists(log_path)

        try:
            cmd = "cat {} |grep 'cost :' | wc -l".format(log_path)
            outval = os.popen(cmd).read()
        except Exception as e:
            raise Exception("raise an exception: {}".format(e))

        assert int(outval) == (loop_num + warmup_num)
        summary_json_path = os.path.join(output_parent_path, "{}_summary.json".format(output_dirname))
        with open(summary_json_path,'r',encoding='utf8') as fp:
            json_data = json.load(fp)
            NPU_compute_time_count = json_data["NPU_compute_time"]["count"]
            h2d_num = json_data["H2D_latency"]["count"]
            d2h_num = json_data["D2H_latency"]["count"]
            assert NPU_compute_time_count == loop_num
            assert h2d_num == 1
            assert d2h_num == 1
        shutil.rmtree(output_path)
        os.remove(summary_json_path)

    def test_pure_inference_normal_pure_data_type(self):
        batch_size = 1
        pure_data_types = ["zero", "random"]
        loop_num = 3
        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "pure_data_type"
        output_path = os.path.join(output_parent_path, output_dirname)
        log_path = os.path.join(output_path, "log.txt")
        for pure_data_type in pure_data_types:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            cmd = "{} --model {} --device {}  --debug true --output {}  --output_dirname {} --pure_data_type {} --loop {} > {}".format(TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, output_parent_path, output_dirname, pure_data_type, loop_num, log_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

            bin_files = []
            for output_file in os.listdir(output_path):
                if output_file.endswith('json') or output_file.endswith('txt'):
                    continue
                bin_files.append(output_file)

            first_output_bin_file_path = os.path.join(output_path, bin_files[0])
            for i, bin_file in enumerate(bin_files):
                bin_file_path = os.path.join(output_path, bin_file)
                if i > 0:
                    if pure_data_type == "zero":
                        assert filecmp.cmp(first_output_bin_file_path, bin_file_path)
                    else:
                        assert filecmp.cmp(first_output_bin_file_path, bin_file_path) == False

        shutil.rmtree(output_path)

    def test_general_inference_prformance_comparison_with_msame_static_batch(self):
        batch_size = 1
        input_file_num = 100
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"), input_file_num)
        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        output_path = os.path.join(self.model_base_path, "output")
        output_dir_name = "ais_infer_output"
        output_dir_path = os.path.join(output_path, output_dir_name)
        if os.path.exists(output_dir_path):
            shutil.rmtree(output_dir_path)
        os.makedirs(output_dir_path)
        summary_json_path = os.path.join(output_path,  "{}_summary.json".format(output_dir_name))

        cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {}".format(TestCommonClass.cmd_prefix, model_path,
                                                                TestCommonClass.default_device_id, input_path, output_path, output_dir_name)
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
        allowable_performance_deviation = 0.03
        reference_deviation = (ais_inference_time_ms - msame_inference_time_ms)/msame_inference_time_ms
        print("static batch msame time:{} ais time:{} ref:{}".format(msame_inference_time_ms, ais_inference_time_ms, reference_deviation))

        assert reference_deviation < allowable_performance_deviation
        os.remove(msame_infer_log_path)
        shutil.rmtree(output_dir_path)

    def test_general_inference_prformance_comparison_with_msame_dynamic_shape(self):
        dym_shape = "actual_input_1:1,3,224,224"
        input_file_num = 100
        output_size = 100000

        model_path = self.get_dynamic_shape_om_path()
        input_size = self.get_dynamic_shape_om_file_size(dym_shape)

        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"), input_file_num)
        output_path = os.path.join(self.model_base_path, "output")
        output_dir_name = "ais_infer_dym_output"
        output_dir_path = os.path.join(output_path, output_dir_name)
        if os.path.exists(output_dir_path):
            shutil.rmtree(output_dir_path)
        os.makedirs(output_dir_path)
        summary_json_path = os.path.join(output_path,  "{}_summary.json".format(output_dir_name))

        cmd = "{} --model {}  --outputSize {} --dymShape {} --input {} --output {} --output_dirname {}".format(
            TestCommonClass.cmd_prefix, model_path,
            output_size, dym_shape, input_path, output_path, output_dir_name)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0

        with open(summary_json_path,'r',encoding='utf8') as fp:
            json_data = json.load(fp)
            ais_inference_time_ms = json_data["NPU_compute_time"]["mean"]

        assert math.fabs(ais_inference_time_ms) > TestCommonClass.EPSILON
        # get msame inference  average time without first time
        msame_infer_log_path = os.path.join(output_path, output_dir_name, "msame_infer.log")
        cmd = "{} --model {} --outputSize {} --dymShape {} --input {} --output {} > {}".format(
            TestCommonClass.msame_bin_path, model_path,
            output_size, dym_shape, input_path, output_path, msame_infer_log_path)
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
        allowable_performance_deviation = 0.04
        reference_deviation = (ais_inference_time_ms - msame_inference_time_ms)/msame_inference_time_ms
        print("dymshape msame time:{} ais time:{} ref:{}".format(msame_inference_time_ms, ais_inference_time_ms, reference_deviation))
        assert reference_deviation < allowable_performance_deviation
        os.remove(msame_infer_log_path)
        shutil.rmtree(output_dir_path)


    def test_pure_inference_batchsize_is_none_normal_static_batch(self):
        """
        batch size 1,2,4,8,16
        """
        batch_list = [1,2,4,8,16]
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_paths = []
        summary_paths = []
        for i, batch_size in enumerate(batch_list):
            output_dirname = "static_batch_{}".format(i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            cmd = "{} --model {} --device {} --output {} --output_dirname {} > {}".format(TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, output_parent_path, output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)
            ret = os.system(cmd)
            assert ret == 0

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert batch_size == cur_batchsize

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

    def test_pure_inference_batchsize_is_none_normal_dynamic_batch(self):
        """
        batch size 1,2,4,8,16
        """
        batch_list = [1,2,4,8]
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_paths = []
        summary_paths = []
        for i, dys_batch_size in enumerate(batch_list):
            output_dirname = "dynamic_batch_{}".format(i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            model_path = self.get_dynamic_batch_om_path()
            cmd = "{} --model {} --device {}  --output {} --output_dirname {} --dymBatch {} > {}".format(TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id,  output_parent_path, output_dirname, dys_batch_size, log_path)
            print("run cmd:{}".format(cmd))
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)
            ret = os.system(cmd)
            assert ret == 0

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

    def test_pure_inference_batchsize_is_none_normal_dynamic_dims(self):
        dym_dims = ["actual_input_1:1,3,224,224", "actual_input_1:8,3,448,448"]
        bs_sizes = [1, 8]
        model_path = self.get_dynamic_dim_om_path()
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_paths = []
        summary_paths = []
        for i, dym_dim in enumerate(dym_dims):
            output_dirname = "dynamic_dims_{}".format(i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            cmd = "{} --model {} --device {} --dymDims {} --output {} --output_dirname {} > \
                {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id, dym_dim,
                           output_parent_path, output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)
            ret = os.system(cmd)
            assert ret == 0

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert bs_sizes[i] == cur_batchsize
                    break

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

    def test_pure_inference_batchsize_is_none_normal_dynamic_hw(self):
        batchsize = 1
        hw_list = ["224,224", "448,448"]
        model_path = self.get_dynamic_hw_om_path()
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_paths = []
        summary_paths = []
        for i, dym_hw in enumerate(hw_list):
            output_dirname = "dynamic_dims_{}".format(i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            cmd = "{} --model {} --device {} --dymHW {} --output {} --output_dirname {} > \
                {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                           dym_hw, output_parent_path, output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)
            ret = os.system(cmd)
            assert ret == 0

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert batchsize == cur_batchsize
                    break

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

    def test_pure_inference_batchsize_is_none_normal_dynamic_shape(self):
        dym_shapes = ["actual_input_1:1,3,224,224", "actual_input_1:1,3,300,300", "actual_input_1:2,3,224,224", "actual_input_1:2,3,300,300",\
            "actual_input_1:4,3,224,224", "actual_input_1:4,3,300,300", "actual_input_1:8,3,300,300", "actual_input_1:8,3,300,300", \
            "actual_input_1:16,3,224,224", "actual_input_1:16,3,300,300"]
        batchsizes = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]
        output_size = 100000
        model_path = self.get_dynamic_shape_om_path()
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_paths = []
        summary_paths = []
        for i, dym_shape in enumerate(dym_shapes):
            output_dirname = "dynamic_shape_{}".format(i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            cmd = "{} --model {} --device {} --outputSize {} --dymShape {} --output {} --output_dirname {} > {}".format(TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, output_size, dym_shape, output_parent_path, output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)
            ret = os.system(cmd)
            assert ret == 0

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert batchsizes[i] == cur_batchsize
                    break

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

    def get_dynamic_shape_range_mode_inference_result_info(self, log_path):
        run_count = 0
        result_OK_num = 0
        shape_status = dict()
        with open(log_path) as f:
            for line in f:
                if 'run_count' in line:
                    str_list = line.split()
                    tmp_str = str_list[1]
                    num_str = tmp_str[(tmp_str.rfind(':') + 1):]
                    num_str = num_str.replace('\n','')
                    run_count = int(num_str)
                if "result:OK throughput" in line:
                    result_OK_num += 1
                    str_list = line.split()
                    tmp_str = str_list[2]
                    shape_str = tmp_str[(tmp_str.find(':') + 1):]
                    shape_str = shape_str.replace('\n','')
                    shape_status[shape_str] = True

        return run_count, result_OK_num,  shape_status

    def test_pure_inference_normal_dynamic_shape_range_mode(self):
        dymShape_range = "actual_input_1:1,3,224,224~226"
        dymshapes = ["actual_input_1:1,3,224,224", "actual_input_1:1,3,224,225", "actual_input_1:1,3,224,226"]
        model_path = self.get_dynamic_shape_om_path()
        output_size = 100000
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "dynamic_shape_range"
        output_path = os.path.join(output_parent_path, output_dirname)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        log_path = os.path.join(output_path, "log.txt")

        cmd = "{} --model {} --outputSize {} --dymShape_range {} --output {} --output_dirname {} > \
            {}".format(TestCommonClass.cmd_prefix, model_path, output_size, dymShape_range, output_parent_path,
                       output_dirname, log_path)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0

        run_count, result_OK_num, shape_status = self.get_dynamic_shape_range_mode_inference_result_info(log_path)
        assert run_count == len(dymshapes)
        assert run_count == result_OK_num
        assert len(dymshapes) == len(shape_status.keys())
        for k, v in shape_status.items():
            assert k in dymshapes
            assert v is True

        shutil.rmtree(output_path)
        os.remove(summary_json_path)

    def test_pure_inference_normal_dynamic_shape_range_mode_2(self):
        dymShape_range = "actual_input_1:1~2,3,224-300,224-300"
        dymshapes = ["actual_input_1:1,3,224,224", "actual_input_1:1,3,224,300", "actual_input_1:1,3,300,224", "actual_input_1:1,3,300,300",
                     "actual_input_1:2,3,224,224", "actual_input_1:2,3,224,300", "actual_input_1:2,3,300,224", "actual_input_1:2,3,300,300"]
        model_path = self.get_dynamic_shape_om_path()
        output_size = 100000
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "dynamic_shape_range"
        output_path = os.path.join(output_parent_path, output_dirname)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        log_path = os.path.join(output_path, "log.txt")

        cmd = "{} --model {} --outputSize {} --dymShape_range {} --output {} --output_dirname {} > \
            {}".format(TestCommonClass.cmd_prefix, model_path, output_size, dymShape_range, output_parent_path,
                       output_dirname, log_path)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0

        run_count, result_OK_num, shape_status = self.get_dynamic_shape_range_mode_inference_result_info(log_path)
        assert run_count == len(dymshapes)
        assert run_count == result_OK_num
        assert len(dymshapes) == len(shape_status.keys())
        for k, v in shape_status.items():
            assert k in dymshapes
            assert v is True

        shutil.rmtree(output_path)
        os.remove(summary_json_path)

    def test_pure_inference_normal_dynamic_shape_range_mode_3(self):
        range_file_parent_path = os.path.join(self.model_base_path,  "input")
        dymShape_range_file = os.path.join(range_file_parent_path, "dymshape_range.info")
        with open(dymShape_range_file, 'w') as f:
            f.write("actual_input_1:1,3,224-300,224-225\n")
            f.write("actual_input_1:8-9,3,224-300,260-300")

        dymshapes = ["actual_input_1:1,3,224,224", "actual_input_1:1,3,224,225", "actual_input_1:1,3,300,224", "actual_input_1:1,3,300,225",
                     "actual_input_1:8,3,224,260", "actual_input_1:8,3,224,300", "actual_input_1:8,3,300,260", "actual_input_1:8,3,300,300",
                     "actual_input_1:9,3,224,260", "actual_input_1:9,3,224,300", "actual_input_1:9,3,300,260", "actual_input_1:9,3,300,300"]
        model_path = self.get_dynamic_shape_om_path()
        output_size = 100000
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "dynamic_shape_range"
        output_path = os.path.join(output_parent_path, output_dirname)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        log_path = os.path.join(output_path, "log.txt")

        cmd = "{} --model {} --outputSize {} --dymShape_range {} --output {} --output_dirname {} > \
            {}".format(TestCommonClass.cmd_prefix, model_path, output_size, dymShape_range_file, output_parent_path,
                       output_dirname,log_path)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0

        run_count, result_OK_num, shape_status = self.get_dynamic_shape_range_mode_inference_result_info(log_path)
        assert run_count == len(dymshapes)
        assert run_count == result_OK_num
        assert len(dymshapes) == len(shape_status.keys())

        for k, v in shape_status.items():
            assert k in dymshapes
            assert v is True

        shutil.rmtree(output_path)
        os.remove(summary_json_path)
        os.remove(dymShape_range_file)

    def test_pure_inference_abnormal_dynamic_shape_range_mode(self):
        dymShape_range = "actual_input_1:1,3~4,224-300,224"
        dymshapes = ["actual_input_1:1,3,224,224", "actual_input_1:1,3,300,224", "actual_input_1:1,4,224,224", "actual_input_1:1,4,300,224"]
        model_path = self.get_dynamic_shape_om_path()
        output_size = 100000
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "dynamic_shape_range"
        output_path = os.path.join(output_parent_path, output_dirname)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        log_path = os.path.join(output_path, "log.txt")
        try:
            cmd = "{} --model {} --outputSize {} --dymShape_range {} --output {} --output_dirname {} > \
                {}".format(TestCommonClass.cmd_prefix, model_path, output_size, dymShape_range, output_parent_path,
                        output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret != 0
        except Exception as e:
            print("some case run failure")

        run_count, result_OK_num, shape_status = self.get_dynamic_shape_range_mode_inference_result_info(log_path)
        assert run_count == len(dymshapes)
        assert 2 == result_OK_num

        shutil.rmtree(output_path)
        os.remove(summary_json_path)

    def get_model_batchsize_from_inference_result(self, log_path):
        batch_size = 0
        if os.path.exists(log_path) is False:
            return batch_size

        key_words = "1000*batchsize"
        with open(log_path) as f:
            for line in f:
                if key_words not in line:
                    continue

                sub_str = line.split('/')[0].split('(')[1].strip(')')
                cur_batchsize = int(sub_str)
                batch_size = int(cur_batchsize)
                break
        return batch_size

    def test_pure_inference_batchsize(self):
        batch_sizes = [1, 2, 4, 8, 16]
        para_batch_size = 16

        output_parent_path = os.path.join(self.model_base_path,  "output")

        output_paths = []
        summary_paths = []
        log_paths = []

        for i, batch_size in enumerate(batch_sizes):
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            output_dirname = "batchsize_{}".format(i)
            output_path = os.path.join(output_parent_path, output_dirname)
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            log_path = os.path.join(output_parent_path, "log_{}.txt".format(i))
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)

            cmd = "{} --model {} --batchsize {} --output {} --output_dirname {} > {}".format(TestCommonClass.cmd_prefix,
                model_path, para_batch_size, output_parent_path, output_dirname, log_path)
            print("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)
            log_paths.append(log_path)

            batch_size_from_reference_result = self.get_model_batchsize_from_inference_result(log_path)
            assert batch_size_from_reference_result > 0
            assert batch_size_from_reference_result == para_batch_size

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)
        for log_path in log_paths:
            os.remove(log_path)

    def test_general_inference_interface_simple(self):
        # interface
        batch_size = 1
        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, TestCommonClass.default_device_id, options)

        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(TestCommonClass.default_device_id)

        outnames = [ session.get_outputs()[0].name ]
        feeds = { session.get_inputs()[0].name : tensor}

        outputs = session.run(outnames, feeds)
        outarray = []
        for out in outputs:
            out.to_host()
            outarray.append(np.array(out))

        # cmd
        input_path = os.path.join(self.model_base_path,  "input", "interface_simple.npy")
        np.save(input_path, ndata)
        infer_sample_output_path = os.path.join(self.model_base_path,  "output", "infer_sample_output.bin")
        out = np.array(outarray)
        out.tofile(infer_sample_output_path)

        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "interface_simple"
        output_path = os.path.join(output_parent_path, output_dirname)
        summary_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        cmd = "{} --model {} --input {} --output {} --output_dirname {} --outfmt BIN".format(TestCommonClass.cmd_prefix,
                    model_path, input_path, output_parent_path, output_dirname)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        output_npy_file_path = os.path.join(output_path, "{}_0.bin".format(output_dirname))

        # compare bin file
        assert filecmp.cmp(infer_sample_output_path, output_npy_file_path)

        shutil.rmtree(output_path)
        os.remove(summary_path)
        os.remove(input_path)
        os.remove(infer_sample_output_path)

    def test_general_inference_interface_dynamicshape(self):
        # interface
        model_path = self.get_dynamic_shape_om_path()
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, TestCommonClass.default_device_id, options)

        session.set_dynamic_shape("actual_input_1:1,3,224,224")
        session.set_custom_outsize([10000])
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(TestCommonClass.default_device_id)

        outnames = [ session.get_outputs()[0].name ]
        feeds = { session.get_inputs()[0].name : tensor}

        outputs = session.run(outnames, feeds)
        print("outputs:", outputs)

        outarray = []
        for out in outputs:
            out.to_host()
            outarray.append(np.array(out))

        # cmd
        infer_dynamicshape_output_path = os.path.join(self.model_base_path,  "output", "infer_dynamicshape_output.bin")
        out = np.array(outarray)
        out.tofile(infer_dynamicshape_output_path)

        dym_shape = "actual_input_1:1,3,224,224"
        output_size = 10000
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "interface_dynamicshape"
        output_path = os.path.join(output_parent_path, output_dirname)
        summary_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        cmd = "{} --model {} --outputSize {} --dymShape {} --output {} --output_dirname {} --outfmt BIN".format(TestCommonClass.cmd_prefix,
                    model_path,  output_size, dym_shape, output_parent_path, output_dirname)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        output_bin_file_path = os.path.join(output_path, "pure_infer_data_0.bin")

        # compare bin file
        assert filecmp.cmp(infer_dynamicshape_output_path, output_bin_file_path)

        shutil.rmtree(output_path)
        os.remove(summary_path)
        os.remove(infer_dynamicshape_output_path)

    def test_general_inference_interface_dynamic_dims(self):
        # interface
        model_path = self.get_dynamic_dim_om_path()
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, TestCommonClass.default_device_id, options)
        dynamic_dims = "actual_input_1:1,3,224,224"
        session.set_dynamic_dims(dynamic_dims)
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)

        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(TestCommonClass.default_device_id)

        outnames = [ session.get_outputs()[0].name ]
        feeds = { session.get_inputs()[0].name : tensor}

        outputs = session.run(outnames, feeds)

        outarray = []
        for out in outputs:
            out.to_host()
            outarray.append(np.array(out))

        # cmd
        infer_dynamic_dims_output_path = os.path.join(self.model_base_path,  "output", "infer_dynamic_dims_output.bin")
        out = np.array(outarray)
        out.tofile(infer_dynamic_dims_output_path)

        model_path = self.get_dynamic_dim_om_path()
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "interface_dynamic_dims"
        output_path = os.path.join(output_parent_path, output_dirname)
        summary_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        cmd = "{} --model {} --device {} --dymDims {} --output {} --output_dirname {} \
            --outfmt BIN".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                 dynamic_dims, output_parent_path, output_dirname)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        output_bin_file_path = os.path.join(output_path, "pure_infer_data_0.bin")

        # compare bin file
        assert filecmp.cmp(infer_dynamic_dims_output_path, output_bin_file_path)

        shutil.rmtree(output_path)
        os.remove(summary_path)
        os.remove(infer_dynamic_dims_output_path)

    def test_general_inference_interface_dynamic_hw(self):
        # interface
        model_path = self.get_dynamic_hw_om_path()
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, TestCommonClass.default_device_id, options)

        session.set_dynamic_hw(224,224)
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(TestCommonClass.default_device_id)

        outnames = [ session.get_outputs()[0].name ]
        feeds = { session.get_inputs()[0].name : tensor}

        outputs = session.run(outnames, feeds)
        outarray = []
        for out in outputs:
            out.to_host()
            outarray.append(np.array(out))

        # cmd
        infer_dynamic_hw_output_path = os.path.join(self.model_base_path,  "output", "infer_dynamic_hw_output.bin")
        out = np.array(outarray)
        out.tofile(infer_dynamic_hw_output_path)
        dym_hw = "224,224"
        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "interface_dynamic_hw"
        output_path = os.path.join(output_parent_path, output_dirname)
        summary_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        cmd = "{} --model {} --device {} --dymHW {} --output {} --output_dirname {} \
            --outfmt BIN".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                 dym_hw, output_parent_path, output_dirname)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        output_bin_file_path = os.path.join(output_path, "pure_infer_data_0.bin")

        # compare bin file
        assert filecmp.cmp(infer_dynamic_hw_output_path, output_bin_file_path)

        shutil.rmtree(output_path)
        os.remove(summary_path)
        os.remove(infer_dynamic_hw_output_path)

    def test_general_inference_interface_dynamic_batchsize(self):
        dys_batch_size = 2
        # interface
        model_path = self.get_dynamic_batch_om_path()
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, TestCommonClass.default_device_id, options)

        session.set_dynamic_batchsize(dys_batch_size)
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(TestCommonClass.default_device_id)

        outnames = [ session.get_outputs()[0].name ]
        feeds = { session.get_inputs()[0].name : tensor}

        outputs = session.run(outnames, feeds)

        outarray = []
        for out in outputs:
            out.to_host()
            outarray.append(np.array(out))

        # cmd
        infer_dynamic_batchsize_output_path = os.path.join(self.model_base_path,  "output", "infer_dynamic_batchsize_output.bin")
        out = np.array(outarray)
        out.tofile(infer_dynamic_batchsize_output_path)

        output_parent_path = os.path.join(self.model_base_path,  "output")
        output_dirname = "interface_dynamic_batchsize"
        output_path = os.path.join(output_parent_path, output_dirname)
        summary_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        cmd = "{} --model {} --device {} --dymBatch {} --output {} --output_dirname {} \
            --outfmt BIN".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
            dys_batch_size, output_parent_path, output_dirname)
        print("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0
        output_bin_file_path = os.path.join(output_path, "pure_infer_data_0.bin")

        # compare bin file
        assert filecmp.cmp(infer_dynamic_batchsize_output_path, output_bin_file_path)

        shutil.rmtree(output_path)
        os.remove(summary_path)
        os.remove(infer_dynamic_batchsize_output_path)

if __name__ == '__main__':
    pytest.main(['test_infer_resnet50.py', '-vs'])
