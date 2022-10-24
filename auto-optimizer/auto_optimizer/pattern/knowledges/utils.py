# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type, Optional
from functools import wraps
import time

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node, PlaceHolder
from auto_optimizer.pattern.pattern import MatchBase


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()
        print(f'func:{func.__name__} args:[{args}, {kwargs}] took: {te - ts: 0.3f}s')
        return res
    return wrapper


def try_access(graph: BaseGraph, name: str, type_: Type[BaseNode] = BaseNode) -> Optional[BaseNode]:
    """
    try access graph node with type check, return None if failed, no exception is thrown
    :param graph: the whole computation graph
    :param name: name of the Node
    :param type_: type of the Node, default to BaseNode, which mean no type check
    :return: Node or None

    NOTE: Some models have duplicate Initializer and PlaceHolder names, if caller didn't specify type_,
    this function will return the Initializer in this case.
    """
    try:
        node = graph.get_value_info(name) if type_ is PlaceHolder else graph[name]
    except KeyError:
        return None
    if not isinstance(node, type_):
        return None
    return node


class NextNodeCount(MatchBase):
    """
    This class constraint matching node to has exactly N next node.
    In practice, this means this node can be merged/sliced/modified/removed without affects other nodes,
    which is a common requirement in computation graph optimization.
    """
    def __init__(self, cnt: int = 1):
        super().__init__()
        self._count = cnt

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if not isinstance(node, (Node, )):
            return False
        if len(node.outputs) != 1:
            return False
        nodes = graph.get_next_nodes(node.outputs[0])
        return len(nodes) == self._count