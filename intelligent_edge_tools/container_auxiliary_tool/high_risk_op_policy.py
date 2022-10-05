# coding: utf-8
import json
import os
from typing import Union

from edge_cbb.edge_global.global_path import edge_global_path
from edge_cbb.log.logger import run_log
from edge_cbb.setting.constant import EdgeOmConstant
from edge_cbb.utils.file_utils import FileUtil, FileCopy
from edge_cbb.utils.result import Result
from glib.utils.high_risk_op_policy import HighRiskOpPolicyDto
from glib.utils.high_risk_op_policy import OpEnumBase

_POLICY: Union[HighRiskOpPolicyDto, None] = None


class SiteHighRiskOp(OpEnumBase):
    """
    SITE 高危特性列表
    """
    CREATE_CONTAINER = "create_container"
    DOWNLOAD_MODEL_FILE = "download_model_file"


class HighRiskConfig:

    def __init__(self):
        self.work_dir = os.path.join(edge_global_path.install_dir, EdgeOmConstant.EDGE_WORK_DIR)
        self.config_file = os.path.join(self.work_dir,
                                        f"{EdgeOmConstant.EDGE_OM_MODULE_NAME}/config/high_risk_ops_policy.json")

    def _create_config_file(self):
        edge_om_path = os.path.join(self.work_dir, EdgeOmConstant.EDGE_OM_MODULE_NAME)
        config_file_dir = os.path.join(edge_om_path, "config")
        allow_all_config_file = os.path.join(config_file_dir, "high_risk_ops_policy_allow_all.json")
        ret = FileCopy.force_copy_file(allow_all_config_file, config_file_dir, "high_risk_ops_policy.json")
        if not ret:
            run_log.error(f"copy allow all high risk config to config_file failed, {ret.error}")
            return ret

        run_log.info("copy allow all high risk config to config_file succeed")
        return Result(result=True, data=os.path.join(config_file_dir, "high_risk_ops_policy.json"))

    def init_high_risk_config(self):
        """
        初始化读取高危特性配置文件
        :return:
        """

        global _POLICY
        if _POLICY:
            return

        ret = self._check_config_file()
        if not ret:
            self.config_file = self._create_config_file().data

        try:
            with open(self.config_file, 'r') as f:
                config_json = json.load(f)
            _POLICY = HighRiskOpPolicyDto.load_from_json(SiteHighRiskOp, config_json)
        except Exception as e:
            run_log.error(f"init policy failed, {e}")
            raise e

    def _check_config_file(self):
        if not FileUtil.check_path_is_exist_and_valid(self.config_file):
            run_log.warning("high risk config file not existed, please create")
            return False

        run_log.info("high risk config_file already exist")
        return True


def _check_allow(srv: SiteHighRiskOp):
    """
    检查高危特性是否开启

    :param srv:
    :return:
    """
    return _POLICY.check_allow(srv)


def is_high_risk_op_allow(srv: SiteHighRiskOp):
    """
    高危特性开关检查

    :param srv:
    :return:
    """
    if not _check_allow(srv):
        run_log.error(f"check high risk op failed, feature {srv.value} is disable")
        return False

    return True
