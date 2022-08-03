import os
import pytest
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
        self.model_name = self.get_model_name(self)
        self.cmd_prefix = TestCommonClass.get_cmd_prefix()
        self.base_path = TestCommonClass.get_basepath()
        self.model_base_path = self.get_model_base_path(self)
        self.output_path = self.get_output_path(self)
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
        return os.path.join(self.base_path, self.model_name)

    def get_output_path(self):
        return os.path.join(self.model_base_path, "output")

    def get_dynamic_batch_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymbatch.om")

    def get_dynamic_hw_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymwh.om")

    def get_dynamic_dim_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymdim.om")

    def get_dynamic_shape_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_resnet50_dymshape.om")

    def get_om_size(self, model_path):
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, self.default_device_id, options)
        intensors_desc = session.get_inputs()
        return intensors_desc[0].realsize

    def test_pure_inference_normal_static_batch(self):
        """
        batch size 1,2,4,8
        """
        batch_list = [1, 2, 4, 8]

        for _, batchsize in enumerate(batch_list):
            model_path = TestCommonClass.get_resnet_static_om_path(batchsize)
            cmd = "{} --model {} --device {}".format(self.cmd_prefix, model_path, self.default_device_id)
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_batch(self):
        batch_list = [1, 2, 4, 8]
        model_path = self.get_dynamic_batch_om_path()
        for _, dys_batch_size in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymBatch {}".format(self.cmd_prefix, model_path, self.default_device_id,
                                                                   dys_batch_size)
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_hw(self):
        batch_list = ["224,224", "448,448"]
        model_path = self.get_dynamic_hw_om_path()
        for _, dym_hw in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymHW {}".format(self.cmd_prefix, model_path, self.default_device_id, dym_hw)
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_dims(self):
        batch_list = ["actual_input_1:1,3,224,224", "actual_input_1:8,3,448,448"]

        model_path = self.get_dynamic_dim_om_path()
        for _, dym_dims in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymDims {}".format(self.cmd_prefix, model_path, self.default_device_id, dym_dims)
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_shape(self):
        dym_shape = "actual_input_1:1,3,224,224"
        output_size = 10000
        model_path = self.get_dynamic_shape_om_path()
        cmd = "{} --model {} --device {} --outputSize {} --dymShape {} ".format(self.cmd_prefix, model_path,
                                                                                self.default_device_id, output_size,
                                                                                dym_shape)
        ret = os.system(cmd)
        assert ret == 0

    def test_general_inference_normal_static_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_resnet_static_om_path(batch_size)
        input_size = self.get_om_size(static_model_path)
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8]

        for _, batchsize in enumerate(batch_list):
            model_path = TestCommonClass.get_resnet_static_om_path(batchsize)
            cmd = "{} --model {} --device {} --input {}".format(self.cmd_prefix, model_path, self.default_device_id,
                                                                input_path)
            ret = os.system(cmd)
            assert ret == 0

    def test_general_inference_normal_dynamic_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_resnet_static_om_path(batch_size)
        input_size = self.get_om_size(static_model_path)
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8]

        for _, dys_batch_size in enumerate(batch_list):
            model_path = self.get_dynamic_batch_om_path()
            cmd = "{} --model {} --device {} --dymBatch {} --input {}".format(self.cmd_prefix, model_path,
                                                                              self.default_device_id, dys_batch_size,
                                                                              input_path)
            ret = os.system(cmd)
            assert ret == 0


if __name__ == '__main__':
    pytest.main(['test_infer_resnet50.py', '-vs'])
