
# Dump config '0|5|10'
TF_DUMP_STEP = '0'

# path to run package operator cmp compare
# default may be /usr/local/Ascend/
CMD_ROOT_PATH = '/usr/local/'

# ASCEND Log Path
ASCEND_LOG_PATH = '/root/ascend/log/plog/'

# TOOL CONFIG
LOG_LEVEL = "NOTSET"
DATA_ROOT_DIR = './precision_data'


# Static dirs, do not change
GRAPH_DIR = DATA_ROOT_DIR + '/graph/'
GRAPH_DIR_ALL = DATA_ROOT_DIR + '/graph/all'
GRAPH_DIR_LAST = DATA_ROOT_DIR + '/graph/last'
GRAPH_DIR_BUILD = DATA_ROOT_DIR + '/graph/json'

FUSION_DIR = DATA_ROOT_DIR + '/fusion/'

DUMP_FILES_NPU = DATA_ROOT_DIR + '/dump/npu/'
DUMP_FILES_OVERFLOW = DATA_ROOT_DIR + '/dump/overflow'
DUMP_FILES_OVERFLOW_DECODE = DATA_ROOT_DIR + '/dump/overflow_decode'
DUMP_FILES_CPU = DATA_ROOT_DIR + '/dump/cpu/'
DUMP_FILES_CPU_LOG = DATA_ROOT_DIR + '/dump/cpu_tf_dump_log.txt'
DUMP_FILES_CPU_NAMES = DATA_ROOT_DIR + '/dump/cpu_tf_tensor_names.txt'
DUMP_FILES_CPU_CMDS = DATA_ROOT_DIR + '/dump/cpu_tf_tensor_cmd.txt'
DUMP_FILES_DECODE = DATA_ROOT_DIR + '/dump/decode/'

VECTOR_COMPARE_PATH = DATA_ROOT_DIR + '/dump/vector/compare/'

# CHECK_OVERFLOW = False

