# coding=utf-8
import os

# Dump config '0|5|10'
TF_DUMP_STEP = '0'

# path to run package operator cmp compare
# default may be /usr/local/Ascend/
CMD_ROOT_PATH = '/usr/local/'

# ASCEND Log Path
ASCEND_LOG_PATH = '/root/ascend/log/plog/'

# TOOL CONFIG
LOG_LEVEL = "NOTSET"
ROOT_DIR = ''

'''
precision_data/
├── npu
│   ├── debug_0
|   |   ├── dump
|   |       └── 20210510101133
|   │   └── graph
|   |       └── ge_proto_00000179_PreRunAfterBuild.txt
│   └── debug_1
├── tf
|   ├── tf_debug
|   └── dump
├── overflow
├── fusion
└── temp
    ├── op_graph
    ├── decode
    |   ├── dump_decode
    |   ├── overflow_decode
    |   └── dump_convert
    └── vector_compare
        ├── 20210510101133
        |   ├── result_123456.csv
        |   └── result_123455.csv
        └── 20210510101134
            └── result_123458.csv
'''

# Static dirs, do not change
DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'precision_data')
# Fusion switch
FUSION_SWITCH_FILE = os.path.join(DATA_ROOT_DIR, 'fusion_switch.cfg')

# fusion
FUSION_DIR = os.path.join(DATA_ROOT_DIR, 'fusion')

# npu dump/graph parent dir
NPU_DIR = os.path.join(DATA_ROOT_DIR, 'npu')
DEFAULT_NPU_DIR = os.path.join(NPU_DIR, 'debug_0')
DEFAULT_NPU_DUMP_DIR = os.path.join(DEFAULT_NPU_DIR, 'dump')
DEFAULT_NPU_GRAPH_DIR = os.path.join(DEFAULT_NPU_DIR, 'graph')

# npu overflow dir
OVERFLOW_DIR = os.path.join(DATA_ROOT_DIR, 'overflow')
NPU_OVERFLOW_DUMP_DIR = os.path.join(OVERFLOW_DIR, 'dump')

# tf dirs
TF_DIR = os.path.join(DATA_ROOT_DIR, 'tf')
TF_DEBUG_DUMP_DIR = os.path.join(TF_DIR, 'tf_debug')
TF_DUMP_DIR = os.path.join(TF_DIR, 'dump')
TF_GRAPH_DIR = os.path.join(TF_DIR, 'graph')

# tmp dirs
TMP_DIR = os.path.join(DATA_ROOT_DIR, 'temp')
OP_GRAPH_DIR = os.path.join(TMP_DIR, 'op_graph')

DECODE_DIR = os.path.join(TMP_DIR, 'decode')
OVERFLOW_DECODE_DIR = os.path.join(DECODE_DIR, 'overflow_decode')
DUMP_DECODE_DIR = os.path.join(DECODE_DIR, 'dump_decode')
DUMP_CONVERT_DIR = os.path.join(DECODE_DIR, 'dump_convert')

VECTOR_COMPARE_PATH = os.path.join(TMP_DIR, 'vector_compare')
TF_TENSOR_NAMES = os.path.join(TMP_DIR, 'tf_tensor_names.txt')
TF_TENSOR_DUMP_CMD = os.path.join(TMP_DIR, 'tf_tensor_cmd.txt')

'''
# graph
GRAPH_DIR = os.path.join(DATA_ROOT_DIR, 'graph')
GRAPH_DIR_ALL = os.path.join(GRAPH_DIR, 'all')
GRAPH_DIR_BUILD = os.path.join(GRAPH_DIR, 'json')
GRAPH_CPU = os.path.join(GRAPH_DIR, 'cpu')

# dump
DUMP_DIR = os.path.join(DATA_ROOT_DIR, 'dump')
DUMP_FILES_NPU = os.path.join(DUMP_DIR, 'npu')
DUMP_FILES_OVERFLOW = os.path.join(DUMP_DIR, 'overflow')
DUMP_FILES_CPU_DEBUG = os.path.join(DUMP_DIR, 'cpu_debug')
DUMP_FILES_CPU = os.path.join(DUMP_DIR, 'cpu')

# dump temp dir
DUMP_TMP_DIR = os.path.join(DUMP_DIR, 'temp')
OP_GRAPH_DIR = os.path.join(DUMP_TMP_DIR, 'op_graph')
DUMP_FILES_OVERFLOW_DECODE = os.path.join(DUMP_TMP_DIR, 'overflow_decode')
DUMP_FILES_CPU_LOG = os.path.join(DUMP_TMP_DIR, 'cpu_tf_dump_log.txt')
DUMP_FILES_CPU_NAMES = os.path.join(DUMP_TMP_DIR, 'cpu_tf_tensor_names.txt')
DUMP_FILES_CPU_CMDS = os.path.join(DUMP_TMP_DIR, 'cpu_tf_tensor_cmd.txt')
DUMP_FILES_DECODE = os.path.join(DUMP_TMP_DIR, 'decode')
DUMP_FILES_CONVERT = os.path.join(DUMP_TMP_DIR, 'convert')
VECTOR_COMPARE_PATH = os.path.join(DUMP_TMP_DIR, 'vector_compare')
'''

# FLAG
PRECISION_TOOL_OVERFLOW_FLAG = 'PRECISION_TOOL_OVERFLOW'
PRECISION_TOOL_DUMP_FLAG = 'PRECISION_TOOL_DUMP'

# DUMP CONFIG
OP_DEBUG_LEVEL = 1
DUMP_GE_GRAPH_VALUE = 2
DUMP_GRAPH_LEVEL_VALUE = 2

# TF_DEBUG
TF_DEBUG_TIMEOUT = 360

# MSACCUCMP
MS_ACCU_CMP = r'msaccucmp.py[c]?'
PYTHON = 'python3'
BUILD_JSON_GRAPH_NAME = 'PreRunAfterBuild'
