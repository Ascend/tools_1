from __future__ import annotations

import logging
import os
from typing import Any, Optional

import aclruntime
from aclruntime import PerfOption, perf
import numpy as np

from ais_bench.infer.backends import BACKEND_REGISTRY, backend
from ais_bench.infer.backends.backend import PerformanceResult

logger = logging.getLogger(__name__)


@BACKEND_REGISTRY.register("acl")
class BackendAcl(backend.Backend):
    def __init__(self, config: Any = None) -> None:
        super().__init__()
        self.model_path = ""
        self.elapsed_ns = 0
        self.inputs = None
        self.batchsize = 1
        self.loop = 1
        self.device_id = 0
        self.options = PerfOption()
        self.parse_config(config)

    @property
    def name(self) -> str:
        return "acl"

    @property
    def model_extension(self) -> str:
        return "om"

    def parse_config(self, config: Any = None) -> None:
        if config is None:
            return
        if config.loop is not None:
            self.loop = config.loop
            self.options.loop = config.loop
        if config.device is not None:
            self.device_id = config.device
            self.options.device_id = config.device

        if config.dymBatch != 0:
            self.options.batchsize = config.dymBatch
            self.options.dynamic_type = 1
        elif config.dymHW is not None:
            width, height = (int(x) for x in config.dymDims.split(','))
            self.options.width = width
            self.options.height = height
            self.options.dynamic_type = 2
        elif config.dymDims is not None:
            self.options.dyn_dims = config.dymDims
            self.options.dynamic_type = 3
        elif config.dymShape is not None:
            self.options.dyn_shapes = config.dymShape
            self.options.dynamic_type = 4

        if config.outputSize is not None:
            self.options.custom_output_size = [
                int(x) for x in config.outputSize.split(',')
            ]
        if config.batchsize is not None:
            self.batchsize = config.batchsize
        if config.acl_json_path is not None:
            self.options.acl_json_path = config.acl_json_path
        if config.jobs is not None:
            self.options.threads = config.jobs

    def load(
        self,
        model_path: str,
        inputs: Optional[list] = None,
        _: Optional[list] = None,
    ) -> BackendAcl:
        if not os.path.exists(model_path):
            raise RuntimeError(f"{model_path} not exist.")
        self.model_path = model_path
        self.inputs = inputs
        return self

    def warm_up(self, dataloader: Iterable, iterations: int = 100) -> None:
        pass

    def predict(self, dataloader: Iterable) -> List[AccuracyResult]:
        pass

    def build(self) -> None:
        pass

    def get_perf(self) -> PerformanceResult:
        ret = PerformanceResult()
        ret.throughput = self.batchsize * self.loop / (self.elapsed_ns / 1e9)
        return ret

    def run(self) -> None:
        inputs = self.inputs
        if self.inputs is None:
            options = aclruntime.session_options()
            options.log_level = 1
            options.loop = 1
            session = aclruntime.InferenceSession(
                self.model_path, self.device_id, options
            )

            def new_tensor(_input_desc):
                ndata = np.frombuffer(bytearray(_input_desc.realsize))
                return aclruntime.BaseTensor(
                    ndata.__array_interface__["data"][0], ndata.nbytes
                )

            inputs = [
                new_tensor(_input) for _input in session.get_inputs()
            ]

        self.elapsed_ns = perf(
            self.model_path,
            inputs,
            options=self.options,
        )
