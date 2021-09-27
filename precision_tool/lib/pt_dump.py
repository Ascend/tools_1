# coding=utf-8
import os
import re
import time
import sys
from lib.util import util
from lib.constant import Constant
import config as cfg
from lib.precision_tool_exception import catch_tool_exception
from lib.precision_tool_exception import PrecisionToolException


class PTDump(object):
    def __init__(self):
        self.log = util.get_log()
        self.npu_files = []
        self.gpu_files = []

    def prepare(self):
        print("test")