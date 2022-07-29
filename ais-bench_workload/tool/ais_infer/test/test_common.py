import os
import sys


class TestCommonClass:
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
