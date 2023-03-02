from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Any, Iterable, Union

import attrs


@attrs.define
class AccuracyResult:
    output: Any = None
    label: Any = None
    prediction: Any = None


@attrs.define
class PerformanceStats:
    min: float = None
    max: float = None
    mean: float = None
    median: float = None
    percentile: float = None


@attrs.define
class PerformanceResult:
    h2d_latency: PerformanceStats = None
    compute_time: PerformanceStats = None
    d2h_latency: PerformanceStats = None
    host_wall_time: float = None
    throughput: float = None


@attrs.define
class InferenceTrace:
    h2d_start: float = None
    h2d_end: float = None
    compute_start: float = None
    compute_end: float = None
    d2h_start: float = None
    d2h_end: float = None


class Backend(ABC):
    """
    Backend interface
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Each of the subclasses must implement this.
        This is called to return the name of backend.
        """

    @property
    def model_extension(self) -> str:
        return "model"

    def initialize(self) -> bool:
        """
        init the resource of backend
        """
        return True

    def finalize(self) -> None:
        """
        release the resource of backend
        """
        pass

    @abstractmethod
    def load(self, model_path: str) -> Backend:
        """
        Each of the subclases must implement this.
        This is called to load a model.
        """

    @abstractmethod
    def warm_up(self, dataloader: Iterable, iterations: int = 100) -> None:
        """
        Each of the subclases must implement this.
        This is called to warmup.
        """

    @abstractmethod
    def predict(
        self, dataloader: Iterable
        ) -> Union[List[AccuracyResult], None]:
        """
        Each of the subclasses must implement this.
        This is called to inference a model
        """

    @abstractmethod
    def build(self) -> None:
        """
        Each of the subclasses must implement this.
        This is called to build a model
        """

    @abstractmethod
    def get_perf(self) -> PerformanceResult:
        """
        Each of the subclasses must implement this.
        This is called to get the performance of the model inference.
        """