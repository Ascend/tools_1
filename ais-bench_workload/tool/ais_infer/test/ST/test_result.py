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
        self.default_device_id = 0
        self._current_dir = os.path.dirname(os.path.realpath(__file__))
        self.cmd_prefix = TestCommonClass.get_cmd_prefix()
        self.base_path = TestCommonClass.get_basepath()

    def test_args_ok(self):
        batchsize = 1
        model_path = TestCommonClass.get_resnet_static_om_path(batchsize)
        cmd = "{} --model {} --device {}".format(self.cmd_prefix, model_path, self.default_device_id)
        cmd = "{} --output {}".format(cmd, self._current_dir)
        ret = os.system(cmd)
        assert ret == 0


if __name__ == '__main__':
    pytest.main(['test_result.py', '-vs'])
