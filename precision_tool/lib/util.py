# coding=utf-8
import csv
import re
import os
import shutil
import numpy as np
import logging
import subprocess
from .precision_tool_exception import PrecisionToolException
import config as cfg

try:
    from rich.traceback import install
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rich_print
    from rich.columns import Columns
    # from rich import print as rich_print
    install()
except ImportError as import_err:
    install = None
    Panel = None
    Table = None
    Columns = None
    rich_print = print
    print("Failed to import rich. some function may disable. Run 'pip3 install rich' to fix it.",
          import_err)

try:
    import readline
    readline.parse_and_bind('tab: complete')
except ImportError as import_error:
    print("Unable to import module: readline. Run 'pip3 install gnureadline pyreadline' to fix it.")

# patterns
# GE_PROTO_BUILD_GRAPH_PATTERN = '^ge_proto.*_Build.*txt$'
GE_PROTO_GRAPH_PATTERN = r'^ge_proto_([0-9]+)_([A-Za-z0-9_-]+)\.txt$'
OFFLINE_DUMP_PATTERN = r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})"
OFFLINE_DUMP_DECODE_PATTERN = \
    r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})\.([a-z]+)\.([0-9]{1,255})\.npy$"
OFFLINE_DUMP_CONVERT_PATTERN = \
    r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})" \
    r"\.([a-z]+)\.([0-9]{1,255})\.([x0-9]+)\.npy$"
OFFLINE_FILE_NAME = 'op_type.op_name.task_id(.stream_id).timestamp'
OP_DEBUG_NAME = 'OpDebug.Node_OpDebug.taskid.timestamp'
CPU_DUMP_DECODE_PATTERN = r"^([A-Za-z0-9_-]+)\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})\.npy$"
CPU_FILE_DECODE_NAME = 'op_name.0(.0).timestamp.npy'
OP_DEBUG_PATTERN = r"Opdebug\.Node_OpDebug\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})"
OP_DEBUG_DECODE_PATTERN = r"Opdebug\.Node_OpDebug\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})\.([a-z]+)\.([0-9]{1,255})\.json"
VECTOR_COMPARE_RESULT_PATTERN = r"result_([0-9]{1,255})\.csv"
TIMESTAMP_DIR_PATTERN = '[0-9]{1,255}'
CSV_SHUFFIX = '.csv'
NUMPY_SHUFFIX = '.npy'
CKPT_META_SHUFFIX = r".*.meta$"


class Util(object):
    def __init__(self):
        self.atc = None
        self.ms_accu_cmp = None
        logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s (%(process)d) -[%(levelname)s]%(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
        self.log = logging.getLogger()

    def get_log(self):
        return self.log

    def execute_command(self, cmd: str):
        """ Execute shell command
        :param cmd: command
        :return: status code
        """
        self.log.debug("[Run CMD]: %s", cmd)
        complete_process = subprocess.run(cmd, shell=True)
        return complete_process.returncode

    @staticmethod
    def empty_dir(dir_path: str) -> bool:
        """ Check if target dir is empty
        :param dir_path: target dir
        :return: bool
        """
        if not os.path.exists(dir_path):
            return True
        if len(os.listdir(dir_path)) == 0:
            return True
        return False

    def convert_proto_to_json(self, proto_file):
        """Convert GE proto graphs to json format.
        command: atc --mode=5 --om=ge_proto_Build.txt --json=xxx.json
        :param proto_file: proto file
        :return: result json file
        """
        src_file = os.path.join(cfg.GRAPH_DIR_ALL, proto_file)
        json_file = proto_file + '.json'
        dst_file = os.path.join(cfg.GRAPH_DIR_BUILD, json_file)
        if os.path.exists(dst_file) and os.path.getmtime(dst_file) > os.path.getmtime(src_file):
            self.log.debug("GE graph build json already exist.")
            return json_file
        cmd = '%s --mode=5 --om=%s --json=%s' % (self._get_atc(), src_file, dst_file)
        self.execute_command(cmd)
        if not os.path.isfile(dst_file):
            self.log.error("Convert GE build graph to json failed. can not find any json file in %s",
                           cfg.GRAPH_DIR_BUILD)
            return None
        self.log.info('Finish convert [%s] build graph from proto to json format.', proto_file)
        return json_file

    def convert_dump_to_npy(self, src_file, dst_path, data_format=''):
        """Convert npu dump files to npy format.
        :param src_file: src file
        :param dst_path: dst path
        :param data_format: target data format
        :return: status code
        """
        self.create_dir(dst_path)
        format_cmd = '' if data_format == '' else '-f %s' % data_format
        cmd = '%s %s convert -d %s -out %s %s' % (cfg.PYTHON, self._get_ms_accu_cmp(), src_file, dst_path, format_cmd)
        return self.execute_command(cmd)

    def compare_vector(self, npu_dump_dir, cpu_dump_dir, graph_json, result_path):
        """Run compare vector command.
        :param npu_dump_dir: npu dump data dir
        :param cpu_dump_dir: cpu dump data dir
        :param graph_json: graph json
        :param result_path: result path
        :return: status code
        """
        self.create_dir(result_path)
        if graph_json is None:
            cmd = '%s %s compare -m %s -f %s -out %s' % (
                cfg.PYTHON, self._get_ms_accu_cmp(), npu_dump_dir, cpu_dump_dir, result_path)
        else:
            cmd = '%s %s compare -m %s -g %s -f %s -out %s' % (
                cfg.PYTHON, self._get_ms_accu_cmp(), npu_dump_dir, cpu_dump_dir, graph_json, result_path)
        return self.execute_command(cmd)

    def list_dump_files(self, path, sub_path=''):
        """List npu dump files in npu dump dir.
        default only list the newest sub dir ordered by timestamp. set sub_path to specific other sub_path
        :param path: dump path
        :param sub_path: sub dir
        :return: dump_files, parent_dirs
        """
        parent_dirs = {}
        dump_files = {}
        newest_sub_path = self._get_newest_dir(path) if sub_path == '' else sub_path
        dump_pattern = re.compile(OFFLINE_DUMP_PATTERN)
        for dir_path, dir_names, file_names in os.walk(os.path.join(path, newest_sub_path), followlinks=True):
            for name in file_names:
                dump_match = dump_pattern.match(name)
                if dump_match is None:
                    continue
                dump_files[name] = self._gen_dump_file_info(name, dump_match, dir_path)
                if dir_path not in parent_dirs:
                    parent_dirs[dir_path] = {}
                parent_dirs[dir_path][name] = dump_files[name]
        return dump_files, parent_dirs

    def list_npu_dump_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, OFFLINE_DUMP_PATTERN, extern_pattern,
                                            self._gen_dump_file_info)

    def list_ge_graph_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, GE_PROTO_GRAPH_PATTERN, extern_pattern,
                                            self._gen_build_graph_file_info)

    def list_npu_dump_decode_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, OFFLINE_DUMP_DECODE_PATTERN, extern_pattern,
                                            self._gen_npu_dump_decode_file_info)

    def list_debug_decode_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, OP_DEBUG_DECODE_PATTERN, extern_pattern,
                                            self._gen_overflow_decode_file_info)

    def list_cpu_dump_decode_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, CPU_DUMP_DECODE_PATTERN, extern_pattern,
                                            self._gen_cpu_dump_decode_file_info)

    def list_cpu_graph_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, CKPT_META_SHUFFIX, extern_pattern,
                                            self._gen_cpu_graph_files_info)

    def list_vector_compare_result_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, VECTOR_COMPARE_RESULT_PATTERN, extern_pattern,
                                            self._gen_vector_compare_result_file_info)

    def list_npu_dump_convert_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, OFFLINE_DUMP_CONVERT_PATTERN, extern_pattern,
                                            self._gen_npu_dump_convert_file_info)

    @staticmethod
    def create_dir(path: str):
        """Create dir if not exist
        :param path: path
        :return: bool
        """
        if os.path.exists(path):
            return True
        try:
            os.makedirs(path, mode=0o700)
        except OSError as err:
            LOG.error("Failed to create %s. %s", path, str(err))
            return False
        return True

    def clear_dir(self, path: str, pattern=''):
        """Clear dir with pattern (file/path name match pattern will be removed)
        :param path: path
        :param pattern: pattern
        :return: None
        """
        if not os.path.exists(path):
            return
        try:
            for f in os.listdir(path):
                if not re.match(pattern, f):
                    continue
                file_path = os.path.join(path, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except OSError as err:
            self.log.error("Failed to remove %s. %s", path, str(err))

    @staticmethod
    def npy_info(path):
        """Get npy information
        :param path: npy path
        :return: (shape, dtype)
        """
        if not str(path).endswith(NUMPY_SHUFFIX):
            raise PrecisionToolException("Npy file [%s] is invalid" % path)
        data = np.load(path, allow_pickle=True)
        return data.shape, data.dtype, data.max(), data.min(), data.mean()

    def gen_npy_info_txt(self, file_name):
        """ Generate numpy info txt.
        :param file_name:
        :return: txt
        """
        shape, dtype, max_data, min_data, mean = self.npy_info(file_name)
        return '[Shape: %s] [Dtype: %s] [Max: %s] [Min: %s] [Mean: %s]' % (shape, dtype, max_data, min_data, mean)

    def print_npy_summary(self, path, file_name, is_convert=False, extern_content=''):
        """Print summary of npy data
        :param path: file path
        :param file_name: file name
        :param is_convert: if convert to txt file
        :param extern_content: extern content append to the summary
        :return: None
        """
        target_file = os.path.join(path, file_name)
        if not os.path.exists(target_file):
            raise PrecisionToolException("File [%s] not exist" % target_file)
        shape, dtype, max_data, min_data, mean = self.npy_info(target_file)
        content = "Shape: %s\nDtype: %s\nMax: %s\nMin: %s\nMean: %s\nPath: %s" % (
            shape, dtype, max_data, min_data, mean, target_file)
        if is_convert:
            content += '\nTxtFile: %s.txt' % target_file
        if extern_content != '':
            content += '\n %s' % extern_content
        self.print_panel(content)
        if is_convert:
            self.save_npy_to_txt(target_file)

    @staticmethod
    def save_npy_to_txt(src_file, dst_file='', align=0):
        """save numpy file to txt file.
        default data will be aligned to the last axis of data.shape
        :param src_file: src file name
        :param dst_file: dst file name
        :param align: data align
        :return: None
        """
        if dst_file == '':
            dst_file = src_file + '.txt'
        data = np.load(src_file)
        shape = data.shape
        data = data.flatten()
        if align == 0:
            if len(shape) == 0:
                align = 1
            else:
                align = shape[-1]
        elif data.size() % align != 0:
            pad_array = np.zeros((align - data.size() % align,))
            data = np.append(data, pad_array)
        np.savetxt(dst_file, data.reshape((-1, align)), delimiter=' ', fmt='%g')

    def read_csv(self, path):
        """Read csv file to list.
        :param path: csv file path
        :return: list
        """
        if not str(path).endswith(CSV_SHUFFIX):
            self.log.error("csv path [%s] is invalid", path)
            return
        rows = []
        with open(path) as f:
            csv_handle = csv.reader(f)
            for row in csv_handle:
                rows.append(row)
        return rows

    @staticmethod
    def print(content):
        rich_print(content)

    @staticmethod
    def create_table(title, columns):
        if Table is None:
            raise PrecisionToolException("No rich module error.")
        table = Table(title=title)
        for column_name in columns:
            table.add_column(column_name, overflow='fold')
        return table

    @staticmethod
    def create_columns(content):
        if Columns is None:
            raise PrecisionToolException("No rich module error.")
        return Columns(content)

    def print_panel(self, content, title='', fit=True):
        """ Print panel.
        :param content: content
        :param title: title
        :param fit: if panel size fit the content
        :return:Node
        """
        if Panel is None:
            print(content)
            return
        if fit:
            self.print(Panel.fit(content, title=title))
        else:
            self.print(Panel(content, title=title))

    @staticmethod
    def _detect_file(file_name, root_dir):
        """Find file in root dir"""
        result = []
        for dir_path, dir_names, file_names in os.walk(root_dir, followlinks=True):
            for name in file_names:
                if re.match(file_name, name):
                    result.append(os.path.join(dir_path, name))
        return result

    def _detect_file_if_not_exist(self, target_file):
        """Find specific file in cmd root path"""
        self.log.info("Try to auto detect file with name: %s.", target_file)
        res = self._detect_file(target_file, cfg.CMD_ROOT_PATH)
        if len(res) == 0:
            raise PrecisionToolException("Cannot find any file named %s in dir %s" % (target_file, cfg.CMD_ROOT_PATH))
        self.log.info("Detect [%s] success. %s", target_file, res)
        return res[0]

    def _get_atc(self):
        if self.atc is None:
            self.atc = self._detect_file_if_not_exist('^atc$')
            # os.environ['PATH'] = os.environ['PATH'] + ':' + atc_path
        return self.atc

    def _get_ms_accu_cmp(self):
        if self.ms_accu_cmp is None:
            self.ms_accu_cmp = self._detect_file_if_not_exist(cfg.MS_ACCU_CMP)
        return self.ms_accu_cmp

    def _get_newest_dir(self, path: str):
        """Find the newest subdir in specific path, subdir should named by timestamp."""
        if not os.path.isdir(path):
            self.log.warning("Path [%s] not exists", path)
            return ''
        paths = os.listdir(path)
        sub_paths = []
        for p in paths:
            if re.match(TIMESTAMP_DIR_PATTERN, p):
                sub_paths.append(p)
        if len(sub_paths) == 0:
            self.log.debug("Path [%s] has no timestamp dirs.", path)
            return ''
        newest_sub_path = sorted(sub_paths)[-1]
        self.log.info("Sub path num:[%d]. Dump dirs[%s], choose[%s]", len(sub_paths), str(sub_paths), newest_sub_path)
        return newest_sub_path

    @staticmethod
    def _list_file_with_pattern(path, pattern, extern_pattern, gen_info_func):
        if path is None or not os.path.exists(path):
            raise PrecisionToolException("Path %s not exist." % path)
        file_list = {}
        re_pattern = re.compile(pattern)
        for dir_path, dir_names, file_names in os.walk(path, followlinks=True):
            for name in file_names:
                print(name)
                match = re_pattern.match(name)
                if match is None:
                    continue
                if extern_pattern != '' and not re.match(extern_pattern, name):
                    continue
                file_list[name] = gen_info_func(name, match, dir_path)
        return file_list

    @staticmethod
    def _gen_build_graph_file_info(name, match, dir_path):
        return {
            "file_name": name,
            "path": os.path.join(dir_path, name),
            "graph_id": int(match.group(1)),
            "graph_name": match.group(2)
        }

    @staticmethod
    def _gen_dump_file_info(name, match, dir_path):
        return {
            "file_name": name,
            "op_name": match.group(2),
            "op_type": match.group(1),
            "task_id": int(match.group(3)),
            "dir_path": dir_path,
            "path": os.path.join(dir_path, name),
            "timestamp": int(match.groups()[-1])
        }

    @staticmethod
    def _gen_npu_dump_decode_file_info(name, match, dir_path):
        return {
            "file_name": name,
            "op_type": match.group(1),
            "op_name": match.group(2),
            "task_id": int(match.group(3)),
            "type": match.groups()[-2],
            "idx": int(match.groups()[-1]),
            "timestamp": int(match.groups()[-3]),
            "dir_path": dir_path,
            "path": os.path.join(dir_path, name)
        }

    @staticmethod
    def _gen_cpu_dump_decode_file_info(name, match, dir_path):
        return {
            "file_name": name,
            "op_name": match.group(1),
            "idx": int(match.group(2)),
            "path": os.path.join(dir_path, name),
            "dir_path": dir_path
        }

    @staticmethod
    def _gen_cpu_graph_files_info(name, match, dir_path):
        return {
            "file_name": name,
            "path": os.path.join(dir_path, name),
            "dir_path": dir_path,
            "timestamp": os.path.getmtime(os.path.join(dir_path, name))
        }

    @staticmethod
    def _gen_overflow_decode_file_info(name, match, dir_path):
        # return FileDesc(file_name=name, task_id=int(match.group(1)), anchor_type=match.groups()[-2],
        #                idx=int(match.groups()[-1]), dir_path=dir_path, path=os.path.join(dir_path, name),
        #                timestamp=int(match.groups()[-3]))
        return {
            "file_name": name,
            "dir_path": dir_path,
            'task_id': int(match.group(1)),
            "path": os.path.join(dir_path, name),
            "type": match.groups()[-2],
            "idx": match.groups()[-1],
            "timestamp": int(match.groups()[-3])
        }

    @staticmethod
    def _gen_vector_compare_result_file_info(name, match, dir_path):
        # return FileDesc(file_name=name, dir_path=dir_path, path=os.path.join(dir_path, name),
        #                timestamp=int(match.group(1)))
        return {
            "file_name": name,
            "dir_path": dir_path,
            "path": os.path.join(dir_path, name),
            "timestamp": int(match.group(1))
        }

    @staticmethod
    def _gen_npu_dump_convert_file_info(name, match, dir_path):
        return {
            "file_name": name,
            "op_type": match.group(1),
            "op_name": match.group(2),
            "task_id": int(match.group(3)),
            "type": match.groups()[-3],
            "idx": int(match.groups()[-2]),
            "timestamp": int(match.groups()[-4]),
            "dir_path": dir_path,
            "path": os.path.join(dir_path, name)
        }


util = Util()
