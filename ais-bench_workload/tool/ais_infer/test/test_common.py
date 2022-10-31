import math
import os
import random
import shutil
import sys

import aclruntime
import numpy as np


class TestCommonClass:
    default_device_id = 0
    cmd_prefix = sys.executable + " " + os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ais_infer.py")
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test/testdata")

    @staticmethod
    def get_cmd_prefix():
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        return sys.executable + " " + os.path.join(_current_dir, "../ais_infer.py")

    @staticmethod
    def get_basepath():
        """
        test/testdata
        """
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(_current_dir, "../test/testdata")

    @staticmethod
    def create_inputs_file(input_path, size, pure_data_type=random, file_name_prefix=""):
        file_path = os.path.join(input_path, "{}{}.bin".format(file_name_prefix, size))
        if pure_data_type == "zero":
            lst = [0 for _ in range(size)]
        else:
            lst = [random.randrange(0, 256) for _ in range(size)]
        barray = bytearray(lst)
        ndata = np.frombuffer(barray, dtype=np.uint8)
        ndata.tofile(file_path)
        return file_path

    @classmethod
    def get_inputs_path(cls, size, input_path, input_file_num, pure_data_type=random):
        """generate input files
        folder structure as follows.
        test/testdata/resnet50/input
                        |_ 196608           # size
                            |- 196608.bin   # base_size_file
                            |_ 5            # input_file_num
        """
        size_path = os.path.join(input_path,  str(size))
        if not os.path.exists(size_path):
            os.makedirs(size_path)

        base_size_file_path = os.path.join(size_path, "{}.bin".format(size))
        if not os.path.exists(base_size_file_path):
            cls.create_inputs_file(size_path, size, pure_data_type)

        size_folder_path = os.path.join(input_path, str(input_file_num))

        if os.path.exists(size_folder_path):
            if len(os.listdir(size_folder_path)) == input_file_num:
                return size_folder_path
            else:
                shutil.rmtree(size_folder_path)

        # create soft link to base_size_file
        os.mkdir(size_folder_path)
        strs = []
        for i in range(input_file_num):
            file_name = "{}-{}.bin".format(size, i)
            file_path = os.path.join(size_folder_path, file_name)
            strs.append("ln -s {} {}".format(base_size_file_path, file_path))

        cmd = ';'.join(strs)
        os.system(cmd)

        return size_folder_path

    @classmethod
    def get_bert_inputs_path(cls, size, input_path, input_file_num, pure_data_type=random):
        """generate input files
        folder structure as follows.
        test/testdata/bert/input          # input_path
                        |_ 1536           # size_path
                            |- input_ids_1536.bin     # input_ids_base_size_file_path
                            |- input_mask_1536.bin    # input_mask_base_size_file_path
                            |- segment_ids_3072.bin   # segment_ids_base_size_file_path
                            |_ 5                      # input_file_num_path_in_size_folder_path
        """
        input_size = size
        size_path = os.path.join(input_path,  str(size))
        input_file_num_path_in_size_folder_path = os.path.join(size_path, str(input_file_num))
        if not os.path.exists(input_file_num_path_in_size_folder_path):
            os.makedirs(input_file_num_path_in_size_folder_path)

        input_ids_base_size_file_path = os.path.join(size_path, "input_ids_{}.bin".format(size))
        input_mask_base_size_file_path = os.path.join(size_path, "input_mask_{}.bin".format(size))
        segment_ids_base_size_file_path = os.path.join(size_path, "segment_ids_{}.bin".format(size))

        if not os.path.exists(input_ids_base_size_file_path):
            input_ids_base_size_file_path = cls.create_inputs_file(size_path, input_size, pure_data_type, "input_ids_")

        if not os.path.exists(input_mask_base_size_file_path):
            input_mask_base_size_file_path = cls.create_inputs_file(size_path, input_size, pure_data_type, "input_mask_")

        if not os.path.exists(segment_ids_base_size_file_path):
            segment_ids_base_size_file_path = cls.create_inputs_file(size_path, input_size, pure_data_type, "segment_ids_")

        size_folder_sub_input_ids_dir_path = os.path.join(input_file_num_path_in_size_folder_path, "input_ids")
        size_folder_sub_input_mask__dir_path = os.path.join(input_file_num_path_in_size_folder_path, "input_mask")
        size_folder_sub_segment_ids_dir_path = os.path.join(input_file_num_path_in_size_folder_path, "segment_ids")

        if os.path.exists(input_file_num_path_in_size_folder_path):
            if os.path.exists(size_folder_sub_input_ids_dir_path) and len(os.listdir(size_folder_sub_input_ids_dir_path)) == input_file_num \
                and os.path.exists(size_folder_sub_input_mask__dir_path) and len(os.listdir(size_folder_sub_input_mask__dir_path)) and \
                os.path.exists(size_folder_sub_segment_ids_dir_path) and len(os.listdir(size_folder_sub_segment_ids_dir_path)):
                return input_file_num_path_in_size_folder_path
            else:
                shutil.rmtree(input_file_num_path_in_size_folder_path)

        os.makedirs(size_folder_sub_input_ids_dir_path)
        os.makedirs(size_folder_sub_input_mask__dir_path)
        os.makedirs(size_folder_sub_segment_ids_dir_path)
        # create soft link to input_ids_base_size_file_path, input_mask_base_size_file_path, segment_ids_base_size_file_path
        cmd_strs = []
        for i in range(input_file_num):
            input_ids_file_name = "input_ids_{}-{}.bin".format(input_size, i)
            input_ids_file_path = os.path.join(size_folder_sub_input_ids_dir_path, input_ids_file_name)
            cmd_strs.append("ln -s {} {}".format(input_ids_base_size_file_path, input_ids_file_path))

            input_mask_file_name = "input_mask_{}-{}.bin".format(input_size, i)
            input_mask_file_path = os.path.join(size_folder_sub_input_mask__dir_path, input_mask_file_name)
            cmd_strs.append("ln -s {} {}".format(input_mask_base_size_file_path, input_mask_file_path))

            segment_ids_file_name = "segment_ids_{}-{}.bin".format(input_size, i)
            segment_ids_file_path = os.path.join(size_folder_sub_segment_ids_dir_path, segment_ids_file_name)
            cmd_strs.append("ln -s {} {}".format(segment_ids_base_size_file_path, segment_ids_file_path))

        cmd = ';'.join(cmd_strs)
        os.system(cmd)

        return input_file_num_path_in_size_folder_path

    @classmethod
    def get_model_static_om_path(cls, batchsize, modelname):
        base_path = cls.get_basepath()
        return os.path.join(base_path, "{}/model".format(modelname), "pth_{}_bs{}.om".format(modelname, batchsize))

    @staticmethod
    def prepare_dir(target_folder_path):
        if os.path.exists(target_folder_path):
            shutil.rmtree(target_folder_path)
        os.makedirs(target_folder_path)

    @staticmethod
    def get_model_inputs_size(model_path):
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, TestCommonClass.default_device_id, options)
        return [meta.realsize for meta in session.get_inputs()]

    @staticmethod
    def get_inference_execute_num(log_path):
        if not os.path.exists(log_path) and not os.path.isfile(log_path):
            return 0

        try:
            cmd = "cat {} |grep 'cost :' | wc -l".format(log_path)
            outval = os.popen(cmd).read()
        except Exception as e:
            raise Exception("grep action raises raise an exception: {}".format(e))
            return 0

        return int(outval.replace('\n', ''))
