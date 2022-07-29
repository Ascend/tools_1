import os
import sys
import pytest
import random
import numpy as np
import aclruntime
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
        self.default_device_id = 0
        self._current_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_name = self.get_model_name(self)
        self.cmd_prefix = TestCommonClass.get_cmd_prefix()
        self.basepath = TestCommonClass.get_basepath()
        self.model_test_path = self.get_model_test_path(self)
        self.model_base_path = os.path.join(self.model_test_path, "model")
        self.aipp_model_base_size = self.get_aipp_model_basesize(self)
        self.no_aipp_model_base_size = self.get_no_aipp_model_basesize(self)
        self.output_path = self.get_output_path(self)
        self.output_file_num = 5
        self.static_batch_size = 1

    def get_model_name(self):
        return "resnet50"

    def get_model_test_path(self):
        """
        supported model names as resnet50, resnet101,...。folder struct as follows
        testdata
        ├── resnet101
        │   ├── input
        │   ├── model
        │   └── output
        ├── resnet50
            ├── input
            ├── model
            └── output
        """
        return os.path.join(self.basepath, self.model_name)

    def get_output_path(self):
        return os.path.join(self.model_test_path, "output")

    def get_aipp_model_basesize(self):
        return ""

    def get_no_aipp_model_basesize(self):
        return ""

    def get_static_om_path(self, batchsize):
        return os.path.join(self.model_base_path, "pth_resnet50_bs{}.om".format(batchsize))

    def get_inputs_file(self, input_path, size):
        # 如果文件存在 就返回
        file_path = os.path.join(input_path, "{}.bin".format(size))
        lst = [random.randrange(0, 256) for _ in range(size)]
        barray = bytearray(lst)
        ndata = np.frombuffer(barray, dtype=np.uint8)
        ndata.tofile(file_path)
        return file_path

    def get_inputs_path(self, size, input_file_num):
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        input_path = os.path.join(_current_dir, "../testdata/resnet50", "{}".format(size))
        if not os.path.exists(input_path):
            os.makedirs(input_path)

        # first create and other soft link
        self.get_inputs_file(input_path, size)

        base_file_path = os.path.join(input_path, "{}.bin".format(size))
        if input_file_num > 1:
            for i in range(input_file_num - 1):
                file_name = "{}-{}.bin".format(size, i + 1)
                file_path = os.path.join(input_path, file_name)
                if not os.path.exists(file_path):
                    cmd = "cd {}; ln -s {} {}".format(input_path, base_file_path, file_name)
                    os.system(cmd)

        return input_path

    def get_dynamic_batch_om_path(self):
        return os.path.join(self.model_base_path, "pth_resnet50_dymbatch.om")

    def get_dynamic_hw_om_path(self):
        return os.path.join(self.model_base_path, "pth_resnet50_dymwh.om")

    def get_dynamic_dim_om_path(self):
        return os.path.join(self.model_base_path, "pth_resnet50_dymdim.om")

    def get_dynamic_shape_om_path(self):
        return os.path.join(self.model_base_path, "pth_resnet50_dymshape.om")

    def get_om_size(self, model_path):
        options = aclruntime.session_options()
        options.log_level = 1
        session = aclruntime.InferenceSession(model_path, self.default_device_id, options)

        intensors_desc = session.get_inputs()
        return intensors_desc[0].realsize

    def test_pure_inference_normal_static_batch(self):
        """
        batch size 1,2,4,8
        """
        batch_list = [1, 2, 4, 8]
        ret = 0
        for _, batchsize in enumerate(batch_list):
            model_path = self.get_static_om_path(batchsize)
            cmd = "{} --model {} --device {}".format(self.cmd_prefix, model_path, self.default_device_id)
            cmd = "{} --output {}".format(cmd, self.output_path)
            ret += os.system(cmd)
        assert ret == 0

    def test_pure_inference_normal_dynamic_batch(self):
        batch_list = [1, 2, 4, 8]
        ret = 0
        model_path = self.get_dynamic_batch_om_path()
        for _, dys_batch_size in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymBatch {}".format(self.cmd_prefix, model_path, self.default_device_id, dys_batch_size)
            cmd = "{} --output {}".format(cmd, self.output_path)
            ret += os.system(cmd)

        assert ret == 0

    def test_pure_inference_normal_dynamic_hw(self):
        batch_list = ["224,224", "448,448"]
        ret = 0
        model_path = self.get_dynamic_hw_om_path()
        for _, dym_hw in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymHW {}".format(self.cmd_prefix, model_path, self.default_device_id, dym_hw)
            cmd = "{} --output {}".format(cmd, self.output_path)
            ret += os.system(cmd)

        assert ret == 0

    def test_pure_inference_normal_dynamic_dims(self):
        batch_list = ["actual_input_1:1,3,224,224", "actual_input_1:8,3,448,448"]
        ret = 0
        model_path = self.get_dynamic_dim_om_path()
        for _, dym_dims in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymDims {}".format(self.cmd_prefix, model_path, self.default_device_id, dym_dims)
            cmd = "{} --output {}".format(cmd, self.output_path)
            ret += os.system(cmd)

        assert ret == 0

    def test_pure_inference_normal_dynamic_shape(self):
        dym_shape = "actual_input_1:1,3,224,224"
        output_size = 10000
        model_path = self.get_dynamic_shape_om_path()
        cmd = "{} --model {} --device {} --outputSize {} --dymShape {} ".format(self.cmd_prefix, model_path, self.default_device_id, output_size, dym_shape)
        cmd = "{} --output {}".format(cmd, self.output_path)
        ret = os.system(cmd)

        assert ret == 0

    def test_general_inference_normal_static_batch(self):
        static_model_path = self.get_static_om_path(self.static_batch_size)
        input_size = self.get_om_size(static_model_path)
        input_path = self.get_inputs_path(input_size, self.output_file_num)
        batch_list = [1, 2, 4, 8]
        ret = 0
        for _, batchsize in enumerate(batch_list):
            model_path = self.get_static_om_path(batchsize)
            cmd = "{} --model {} --device {} --input {}".format(self.cmd_prefix, model_path, self.default_device_id, input_path)
            cmd = "{} --output {}".format(cmd, self.output_path)
            ret += os.system(cmd)

        assert ret == 0

    def test_general_inference_normal_dynamic_batch(self):
        static_model_path = self.get_static_om_path(self.static_batch_size)
        input_size = self.get_om_size(static_model_path)
        input_path = self.get_inputs_path(input_size, self.output_file_num)
        batch_list = [1, 2, 4, 8]
        ret = 0
        for _, dys_batch_size in enumerate(batch_list):
            model_path = self.get_dynamic_batch_om_path()
            cmd = "{} --model {} --device {} --dymBatch {} --input {}".format(self.cmd_prefix, model_path, self.default_device_id, dys_batch_size, input_path)
            cmd = "{} --output {}".format(cmd, self.output_path)
            ret += os.system(cmd)

        assert ret == 0


if __name__ == '__main__':
    pytest.main(['test_infer.py'])
