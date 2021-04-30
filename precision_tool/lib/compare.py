# coding=utf-8
"""
Compare
"""
import os
import numpy as np
from lib.tool_object import ToolObject
from lib.graph import Graph
from lib.dump import Dump
from lib.util import util
import config as cfg
from lib.precision_tool_exception import PrecisionToolException
from lib.precision_tool_exception import catch_tool_exception

graph = Graph()
dump = Dump()
NEW_LINE = '\n'

ROW_MAP = {
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
}

ROW_MAP_DESC = r"""[CosSim: CosineSimilarity]
[MaxAbs: MaxAbsoluteError]
[ARE: MaxAbsoluteError]
[RED: RelativeEuclideanDistance]
[KLD: KullbackLeiblerDivergence]"""


class Compare(ToolObject):
    def __init__(self):
        """Init"""
        super(Compare, self).__init__()
        self.log = util.get_log()
        self.vector_compare_result = None

    @catch_tool_exception
    def prepare(self):
        util.create_dir(cfg.VECTOR_COMPARE_PATH)
        self._parse_result_files()

    def vector_compare(self, lh_path, rh_path, graph_json=None):
        """Compare all ops"""
        if lh_path is None or util.empty_dir(lh_path):
            raise PrecisionToolException("No valid dump file in %s" % lh_path)
        if rh_path is None or util.empty_dir(rh_path):
            raise PrecisionToolException("No valid dump file in %s" % rh_path)
        self.log.info("Start vector compare process...")
        util.compare_vector(lh_path, rh_path, graph_json, cfg.VECTOR_COMPARE_PATH)
        self.log.info("Vector compare process finish.")
        self._parse_result_files()
        self.vector_summary()

    @catch_tool_exception
    def vector_summary(self, file_name=None, cos_sim_threshold=0.98):
        """Print not NaN result in vector compare result"""
        self._parse_result_files()
        if self.vector_compare_result is None or len(self.vector_compare_result) == 0:
            raise PrecisionToolException("Can not find any vector compare result in dir:%s" % cfg.VECTOR_COMPARE_PATH)
        if file_name is None:
            # find the latest result
            file_list = sorted(self.vector_compare_result.values(), key=lambda x: x['timestamp'])
            file_names = [x['file_name'] for x in file_list]
            file_name = file_names[-1]
            self.log.debug("Find %s result files. Choose [%s]", file_names, file_name)
        if file_name not in self.vector_compare_result:
            raise PrecisionToolException("Can not find file:%s in dir:%s" % (file_name, cfg.VECTOR_COMPARE_PATH))
        file_info = self.vector_compare_result[file_name]
        items = util.read_csv(file_info['path'])
        # table = util.create_table(file_name, list(ROW_MAP.keys())[3:])
        table = util.create_table(file_name, ['Index', 'TensorIndex', 'CosineSimilarity'])
        for item in items:
            if len(item) == 0 or item[ROW_MAP['TensorIdx']] == 'TensorIndex' or item[ROW_MAP['MaxAbs']] == 'NaN':
                continue
            # valid data
            if float(item[ROW_MAP['CosSim']]) < cos_sim_threshold:
                table.add_row(item[ROW_MAP['Index']], item[ROW_MAP['TensorIdx']], item[ROW_MAP['CosSim']])
                # table.add_row(item[ROW_MAP['TensorIdx']], item[ROW_MAP['CosSim']], item[ROW_MAP['MaxAbs']],
                #              item[ROW_MAP['ARE']], item[ROW_MAP['RED']], item[ROW_MAP['KLD']],
                #              item[ROW_MAP['StandardDeviation']])
                break

        util.print(table)
        util.print(ROW_MAP_DESC)

    def compare_data(self, left, right, save_txt=False, rl=0.001, al=0.001, diff_count=20):
        """Compare data"""
        left = self._detect_file(left)
        right = self._detect_file(right)
        if left is None or right is None:
            raise PrecisionToolException("invalid input or output")
        # save to txt
        if save_txt:
            util.save_npy_to_txt(left)
            util.save_npy_to_txt(right)
        # compare data
        total_cnt, all_close, cos_sim, err_percent = self._do_compare_data(left, right, rl, al, diff_count)
        content = ['Left:', ' |- NpyFile: %s' % left]
        if save_txt:
            content.append(' |- TxtFile: [green]%s.txt[/green]' % left)
        content.append(' |- NpySpec: [yellow]%s[/yellow]' % util.gen_npy_info_txt(left))
        content.append('DstFile:')
        content.append(' |- NpyFile: %s' % right)
        if save_txt:
            content.append(' |- TxtFile: [green]%s.txt[/green]' % right)
        content.append(' |- NpySpec: [yellow]%s[/yellow]' % util.gen_npy_info_txt(right))
        content.append('NumCnt:   %s' % total_cnt)
        content.append('AllClose: %s' % all_close)
        content.append('CosSim:   %s' % cos_sim)
        content.append('ErrorPer: %s  (rl= %s, al= %s)' % (err_percent, rl, al))
        util.print_panel(NEW_LINE.join(content))

    def _do_compare_data(self, left, right, rl=0.001, al=0.001, diff_count=20):
        data_left = np.load(left).astype(np.float32)
        data_right = np.load(right).astype(np.float32)
        shape_left = data_left.shape
        shape_right = data_right.shape
        if shape_left != shape_right:
            self.log.warning("Data shape not equal: %s vs %s", data_left.shape, data_right.shape)
        data_left = data_left.reshape(-1)
        data_right = data_right.reshape(-1)
        if data_left.shape[0] != data_right.shape[0]:
            self.log.warning("Data size not equal: %s vs %s", data_left.shape, data_right.shape)
        all_close = np.allclose(data_left, data_right, atol=al, rtol=rl)
        # cos_sim = 1 - spatial.distance.cosine(data_left, data_right)
        cos_sim = np.dot(data_left, data_right) / (
                np.sqrt(np.dot(data_left, data_left)) * np.sqrt(np.dot(data_right, data_right)))
        err_cnt = 0
        total_cnt = data_left.shape[0]
        diff_table_columns = ['Index', 'Left', 'Right', 'Diff']
        err_table = util.create_table("Error Item Table", diff_table_columns)
        top_table = util.create_table("Top Item Table", diff_table_columns)
        for i in range(total_cnt):
            abs_diff = abs(data_left[i] - data_right[i])
            if i < diff_count:
                top_table.add_row(str(i), str(data_left[i]), str(data_right[i]), str(abs_diff))
            if abs_diff > (al + rl * abs(data_right[i])):
                if err_cnt < diff_count:
                    err_table.add_row(str(i), str(data_left[i]), str(data_right[i]), str(abs_diff))
                err_cnt += 1
        err_percent = float(err_cnt / total_cnt)
        util.print(util.create_columns([err_table, top_table]))
        return total_cnt, all_close, cos_sim, err_percent

    def _parse_result_files(self):
        self.vector_compare_result = util.list_vector_compare_result_files(cfg.VECTOR_COMPARE_PATH)
        if len(self.vector_compare_result) == 0:
            raise PrecisionToolException("Can not find any vector compare result in dir:%s" % cfg.VECTOR_COMPARE_PATH)
        self.log.info("Find vector compare result files. %s", self.vector_compare_result.keys())

    @staticmethod
    def _detect_file(file_name):
        """Find files in npu/overflow/cpu dump dir"""
        if os.path.isfile(file_name):
            return file_name
        for parent_dir in [cfg.DUMP_FILES_DECODE, cfg.DUMP_FILES_CPU, cfg.DUMP_FILES_OVERFLOW_DECODE,
                           cfg.DUMP_FILES_CONVERT]:
            if os.path.isfile(os.path.join(parent_dir, file_name)):
                return os.path.join(parent_dir, file_name)
        return None
