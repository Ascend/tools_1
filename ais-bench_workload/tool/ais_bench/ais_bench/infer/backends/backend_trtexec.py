from __future__ import annotations

import os
import sys
import logging
import subprocess
import re
from typing import Iterable, List, Dict, Any

from ais_bench.infer.backends import backend, BACKEND_REGISTRY
from ais_bench.infer.backends.backend import AccuracyResult, PerformanceStats, PerformanceResult, InferenceTrace


class TrtexecConfig(object):
    def __init__(self):
        self.iterations = None
        self.warmUp = None
        self.duration = None
        self.batch = None
        self.device = None


logger = logging.getLogger(__name__)


@BACKEND_REGISTRY.register("trtexec")
class BackendTRTExec(backend.Backend):
    def __init__(self, config: Any = None) -> None:
        super(BackendTRTExec, self).__init__()
        self.config = TrtexecConfig()
        self.convert_config(config)
        self.model_path = ""
        self.output_log = ""
        self.trace = InferenceTrace()

    @property
    def name(self) -> str:
        return "trtexec"

    @property
    def model_extension(self) -> str:
        return "plan"

    def convert_config(self, config):
        if config.loop != None:
            self.config.iterations = config.loop
        if config.warmup_count != None:
            self.config.warmup_count = config.warmup_count
        if config.batchsize != None:
            self.config.batch = config.batchsize
        if config.device != None:
            self.config.device = config.device

    def load(
        self, model_path: str, inputs: list = None, outputs: list = None
    ) -> BackendTRTExec:
        if os.path.exists(model_path):
            logger.info("Load engine from file {}".format(model_path))
            self.model_path = model_path
        else:
            raise Exception("{} not exit".format(model_path))
        return self

    def parse_perf(self, data: List) -> PerformanceStats:
        stats = PerformanceStats()
        stats.min = float(data[0])
        stats.max = float(data[1])
        stats.mean = float(data[2])
        stats.median = float(data[3])
        stats.percentile = float(data[4])
        return stats

    def parse_log(self, log: str) -> PerformanceResult:
        performance = PerformanceResult()
        log_list = log.splitlines()
        pattern_1 = re.compile(r"(?<=: )\d+\.?\d*")
        pattern_2 = re.compile(r"(?<== )\d+\.?\d*")
        for line in log_list:
            if "Throughput" in line:
                throughput = pattern_1.findall(line)
                performance.throughput = float(throughput[0])
            elif "H2D Latency" in line:
                h2d_latency = pattern_2.findall(line)
                performance.h2d_latency = self.parse_perf(h2d_latency)
            elif "GPU Compute Time: min" in line:
                compute_time = pattern_2.findall(line)
                performance.compute_time = self.parse_perf(compute_time)
            elif "D2H Latency" in line:
                d2h_latency = pattern_2.findall(line)
                performance.d2h_latency = self.parse_perf(d2h_latency)
            elif "Total Host Walltime" in line:
                total_host_time = pattern_1.findall(line)
                performance.host_wall_time = float(total_host_time[0])
        return performance

    def warm_up(self, dataloader: Iterable, iterations: int = 100) -> None:
        pass

    def predict(self, dataloader: Iterable) -> List[AccuracyResult]:
        pass

    def build(self) -> None:
        pass

    def get_perf(self) -> PerformanceResult:
        return self.parse_log(self.output_log)

    def run(self):
        command = [
            "trtexec",
            f"--onnx={self.model_path}",
            f"--fp16",
        ]
        if self.config.duration != None:
            command.append(f"--duration={self.config.duration}")
        if self.config.device != None:
            command.append(f"--device={self.config.device}")
        if self.config.iterations != None:
            command.append(f"--iterations={self.config.iterations}")
        if self.config.warmUp != None:
            command.append(f"--warmUp={self.config.warmUp}")
        if self.config.batch != None:
            command.append(f"--batch={self.config.batch}")

        logger.info("Trtexec Build command: " + " ".join(command))
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, shell=False
        )

        while process.poll() is None:
            line = process.stdout.readline()
            self.output_log += line.decode()
            line = line.strip()
            if line:
                print(line.decode())

        return []
