import os
import sys

import pytest
from test_common import TestCommonClass


class TestClass:
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
        self.model_name = "resnet50"

    def test_args_ok(self):
        output_path = os.path.join(TestCommonClass.base_path, "tmp")
        TestCommonClass.prepare_dir(output_path)
        model_path = TestCommonClass.get_model_static_om_path(2, self.model_name)
        cmd = "{} --model {} --device {}".format(TestCommonClass.cmd_prefix, model_path,
                                                 TestCommonClass.default_device_id)
        cmd = "{} --output {}".format(cmd, output_path)
        ret = os.system(cmd)
        assert ret == 0


if __name__ == '__main__':
    pytest.main(['test_result.py', '-vs'])
