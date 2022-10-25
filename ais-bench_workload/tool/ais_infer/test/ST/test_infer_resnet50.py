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
        self.auto_set_dymshape_mode_input_dir_path = os.path.join(self.model_base_path, "model", "input", "auto_set_dymshape_mode_input")

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

    def create_npy_files_in_auto_set_dymshape_mode_input(self, shapes):
        # default the folder is not existed
        os.makedirs(self.auto_set_dymshape_mode_input_dir_path)

        i = 1
        for shape in shapes:
            x = np.zeros(shape, dtype=np.int32)
            file_name = 'input_shape_{}'.format(i)
            file = os.path.join(self.auto_set_dymshape_mode_input_dir_path, "{}.npy".format(file_name))
            np.save(file, x)
            i += 1

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
        multiple npy input files as input parameter
        """
        shapes = [[1, 3,  224,  224], [1, 3, 300, 300], [1, 3, 200, 200]]
        if not os.path.exists(self.auto_set_dymshape_mode_input_dir_path):
            self.create_npy_files_in_auto_set_dymshape_mode_input(shapes)
        else:
            filelist = os.listdir(self.auto_set_dymshape_mode_input_dir_path)
            for file_path in filelist:
                if not file_path.endswith(".npy"):
                    os.remove(file_path)
            if len(os.listdir(self.auto_set_dymshape_mode_input_dir_path)) == 0:
                self.create_npy_files_in_auto_set_dymshape_mode_input(shapes)

        output_size = 10000
        model_path = self.get_dynamic_shape_om_path()
        filelist = os.listdir(self.auto_set_dymshape_mode_input_dir_path)
        num_shape = len(filelist)
        file_paths = []
        for file in filelist:
            file_paths.append(os.path.join(self.auto_set_dymshape_mode_input_dir_path, file))
        file_paths = ",".join(file_paths)
        output_parent_path = os.path.join(self.model_base_path, "model", "output")
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

    def test_inference_normal_dynamic_shape_auto_set_dymshape_mode_2(self):
        """"
        a folder containing multiple npy files as input parameter
        """
        shapes = [[1, 3,  224,  224], [1, 3, 300, 300], [1, 3, 200, 200]]
        if not os.path.exists(self.auto_set_dymshape_mode_input_dir_path):
            self.create_npy_files_in_auto_set_dymshape_mode_input(shapes)
        else:
            filelist = os.listdir(self.auto_set_dymshape_mode_input_dir_path)
            for file_path in filelist:
                if not file_path.endswith(".npy"):
                    os.remove(file_path)
            if len(os.listdir(self.auto_set_dymshape_mode_input_dir_path)) == 0:
                self.create_npy_files_in_auto_set_dymshape_mode_input(shapes)

        output_size = 10000
        model_path = self.get_dynamic_shape_om_path()
        filelist = os.listdir(self.auto_set_dymshape_mode_input_dir_path)
        num_shape = len(filelist)
        output_parent_path = os.path.join(self.model_base_path, "model", "output")
        output_dirname = "auto_set_dymshape_mode_output"
        output_path = os.path.join(output_parent_path, output_dirname)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        cmd = "{} --model {} --device {} --outputSize {} --auto_set_dymshape_mode true --input {} --output {}  --output_dirname {} ".format(TestCommonClass.cmd_prefix, model_path,
            TestCommonClass.default_device_id, output_size, self.auto_set_dymshape_mode_input_dir_path, output_parent_path, output_dirname)

        ret = os.system(cmd)
        assert ret == 0

        try:
            cmd = "find {} -name '*.bin'|wc -l".format(output_path)
            bin_num = os.popen(cmd).read()
        except Exception as e:
            raise Exception("raise an exception: {}".format(e))

        assert int(bin_num) == num_shape

    def test_general_inference_normal_static_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
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
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
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
    pytest.main(['test_infer_resnet50.py', '-vs'])
