import collections
import os
from lib.util import util
from lib.precision_tool_exception import PrecisionToolException
from lib.precision_tool_exception import catch_tool_exception


class RowMap(object):
    """
    'Index': 0,
    'LeftOp': 1,
    'RightOp': 2,
    'TensorIdx': 3,    # TensorIndex
    'CosSim': 4,    # CosineSimilarity
    'MaxAbs': 5,   # MaxAbsoluteError
    'ARE': 6,   # AccumulatedRelativeError
    'RED': 7,   # RelativeEuclideanDistance
    'KLD': 8,   # KullbackLeiblerDivergence
    'StandardDeviation': 9     # StandardDeviation
    """
    index = 0
    left = 1
    right = 2
    tensor_index = 3
    cosine_similarity = 4
    max_abs = 5


class CompareItem(object):
    def __init__(self, left, index, right):
        self.left = left
        self.index = int(index)
        self.right = right
        self.input = []
        self.output = []

    def update(self, item):
        tensor_index = item[RowMap.tensor_index]
        if tensor_index not in ['NaN', '']




class CompareResult(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.ops = None

    @catch_tool_exception
    def prepare(self):
        if util.empty_dir(self.file_path):
            raise PrecisionToolException("Compare result file %s not exist" % self.file_path)
        items = util.read_csv(self.file_path)
        self.ops = collections.OrderedDict()
        for item in items:
            op_name = item[RowMap.left]
            if op_name not in self.ops:
                self.ops[op_name] = CompareItem(op_name, item[RowMap.index], item[RowMap.right])
            op = self.ops[op_name]
            op.update(item)


