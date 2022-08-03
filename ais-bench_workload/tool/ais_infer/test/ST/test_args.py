import json
import os
import shutil

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
        self.base_path = TestCommonClass.get_basepath()  # test/testdata
        self.resnet50_model_base_path = os.path.join(self.base_path, 'resnet50/model')

    def test_args_invalid_device_id(self):
        device_id = 100
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        cmd = "{} --model {} --device {}".format(self.cmd_prefix, model_path, device_id)
        ret = os.system(cmd)
        assert ret != 0

    def test_args_invalid_model_path(self):
        model_path = "xxx_invalid.om"
        cmd = "{} --model {} --device {}".format(self.cmd_prefix, model_path, self.default_device_id)
        ret = os.system(cmd)
        assert ret != 0

    def test_args_invalid_acl_json(self):
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        acl_json_path = "xxx_invalid.json"
        cmd = "{} --model {} --device {}".format(self.cmd_prefix, model_path, self.default_device_id)
        cmd = "{} --acl_json_path {}".format(cmd, acl_json_path)
        ret = os.system(cmd)
        assert ret != 0

    def test_args_ok(self):
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        cmd = "{} --model {} --device {}".format(self.cmd_prefix, model_path, self.default_device_id)
        print("cmd: {}".format(cmd))

        with open('./log.txt', 'w') as f:
            print("cmd: {}".format(cmd), file=f)
        ret = os.system(cmd)
        assert ret == 0

    def test_args_loop_ok(self):
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        loop_num = 3
        warmup_num = 5
        log_path = "./log.txt"
        cmd = "{} --model {} --device {} --loop {} --debug True > {}".format(self.cmd_prefix, model_path,
                                                                             self.default_device_id, loop_num, log_path)
        ret = os.system(cmd)
        assert ret == 0

        try:
            cmd = "cat {} |grep 'aclExec const' | wc -l".format(log_path)
            outval = os.popen(cmd).read()
        except IOError:
            raise IOError("read error")

        assert int(outval) == (loop_num + warmup_num)

    def test_args_debug_ok(self):
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        log_path = "./log.txt"
        cmd = "{} --model {} --device {} --debug True > {}".format(self.cmd_prefix, model_path,
                                                                   self.default_device_id, log_path)
        ret = os.system(cmd)
        assert ret == 0

        try:
            cmd = "cat {} |grep '[DEBUG]' | wc -l".format(log_path)
            outval = os.popen(cmd).read()
        except IOError:
            raise IOError("read error")

        assert int(outval) > 1

    def test_args_profiler_ok(self):
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        output = "./"
        profiler_path = os.path.join(output, "profiler")
        pre_sub_folder_num = 0
        if os.path.exists(profiler_path):
            pre_sub_folder_num = len(os.listdir(profiler_path))

        cmd = "{} --model {} --device {} --profiler --output {}".format(self.cmd_prefix, model_path,
                                                                        self.default_device_id, output)
        ret = os.system(cmd)
        assert ret == 139

        assert os.path.exists(profiler_path)
        paths = os.listdir(profiler_path)
        assert len(paths) == pre_sub_folder_num + 1

    def test_args_dump_ok(self):
        """
        dump folder existed. and  a sub-folder named with the format of date and time
        """
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        output = "./"
        log_path = "./log.txt"
        dump_path = os.path.join(output, "dump")
        pre_sub_folder_num = 0
        if os.path.exists(dump_path):
            pre_sub_folder_num = len(os.listdir(dump_path))

        cmd = "{} --model {} --device {} --dump --output {} > {}".format(self.cmd_prefix, model_path,
                                                                         self.default_device_id, output, log_path)
        ret = os.system(cmd)
        assert ret == 0
        assert os.path.exists(dump_path)
        paths = os.listdir(dump_path)
        assert len(paths) == pre_sub_folder_num + 1

    def test_args_output_ok(self):
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        output_path = "./"
        log_path = "./log.txt"
        cmd = "{} --model {} --device {}  --output {} > {}".format(self.cmd_prefix, model_path,
                                                                   self.default_device_id, output_path, log_path)
        ret = os.system(cmd)
        assert ret == 0

        try:
            cmd = "cat {} |grep 'output path'".format(log_path)
            outval = os.popen(cmd).read()
        except IOError:
            raise IOError("read error")

        str_list = outval.split(':')
        assert len(str_list) == 2
        paths = os.listdir(str_list[1].replace('\n', ''))
        assert len(paths) == 2

    def test_args_acljson_ok(self):
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        profiler_path = os.path.join(self._current_dir, "profiler")
        output_json_dict = {"profiler": {"switch": "on", "aicpu": "on", "output": "", "aic_metrics": ""}}
        out_json_file_path = os.path.join(self._current_dir, "acl.json")
        if os.path.exists(profiler_path):
            shutil.rmtree(profiler_path)

        with open(out_json_file_path, "w") as f:
            json.dump(output_json_dict, f, indent=4, separators=(", ", ": "), sort_keys=True)
        cmd = "{} --model {} --device {} --acl_json_path {} ".format(self.cmd_prefix, model_path,
                                                                     self.default_device_id, out_json_file_path)
        ret = os.system(cmd)
        assert ret == 139

    def test_args_default_outfmt_ok(self):
        """test default output file suffix case
        there are two output files and one file with bin suffix in output folder path.
        """
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        output_path = "./"
        log_path = "./log.txt"
        output_file_suffix = "BIN"
        cmd = "{} --model {} --device {} --output {} > {}".format(self.cmd_prefix, model_path, self.default_device_id,
                                                                  output_path, log_path)
        ret = os.system(cmd)
        assert ret == 0

        try:
            cmd = "cat {} |grep 'output path'".format(log_path)
            outval = os.popen(cmd).read()
        except IOError:
            raise IOError("test_args_default_outfmt_ok() read error")

        str_list = outval.split(':')
        output_path = os.listdir(str_list[1].replace('\n', ''))
        assert len(output_path) == 2
        num = 0
        for file in output_path:
            if file.split('.')[1] == output_file_suffix.lower():
                num += 1
        assert num > 0

    def test_args_outfmt_ok(self):
        """test supported output file suffix cases
        there are two output files and one file with given suffix in output folder path.
        """
        model_path = TestCommonClass.get_resnet_static_om_path(1)
        output_path = "./"
        log_path = "./log.txt"
        output_file_suffixs = ["NPY", "BIN", "TXT"]

        for _, output_file_suffix in enumerate(output_file_suffixs):
            cmd = "{} --model {} --device {} --output {} --outfmt {} > {}".format(self.cmd_prefix, model_path,
                                                                                  self.default_device_id,
                                                                                  self._current_dir, output_file_suffix,
                                                                                  log_path)
            ret = os.system(cmd)
            assert ret == 0

            try:
                cmd = "cat {} |grep 'output path'".format(log_path)
                outval = os.popen(cmd).read()
            except IOError:
                raise IOError("test_args_outfmt_ok() read error")

            str_list = outval.split(':')
            output_path = os.listdir(str_list[1].replace('\n', ''))
            assert len(output_path) == 2
            num = 0
            for file in output_path:
                if file.split('.')[1] == output_file_suffix.lower():
                    num += 1
            assert num > 0


if __name__ == '__main__':
    pytest.main(['test_args.py', '-vs'])
