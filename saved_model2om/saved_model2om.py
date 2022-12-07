import os
import argparse
import sys
import subprocess
import copy
from time import strftime, localtime
from collections import namedtuple, OrderedDict

import tensorflow as tf
from tensorflow.python.tools import saved_model_cli
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.core.framework import types_pb2, graph_pb2, node_def_pb2, attr_value_pb2
from tensorflow.compat.v1 import graph_util

from six import StringIO
import contextlib
import time
import shutil

NodeInfo = namedtuple('NodeInfo', ['name', 'shape', 'type', 'full_name'])


@contextlib.contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr

    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def get_input_output_node(saved_model_dir, saved_tags, sign):

    parser = saved_model_cli.create_parser()
    saved_model_cli_args = parser.parse_args([
        'show',
        '--dir', saved_model_dir,
        '--tag_set', saved_tags,
        '--signature_def', sign
    ])

    with captured_output() as (out, err):
        saved_model_cli.show(saved_model_cli_args)

    result = out.getvalue().strip()
    print(result)

    input_nodes = OrderedDict()
    output_nodes = OrderedDict()
    lines = result.split('\n')
    for idx, line in enumerate(result.split('\n')):
        if "inputs[" in line:
            parse_node_from_line(idx, input_nodes, line, lines)
        if "outputs[" in line:
            parse_node_from_line(idx, output_nodes, line, lines)

    if not output_nodes:
        raise RuntimeError("No Output Nodes found in saved_model.")

    return input_nodes, output_nodes


def parse_node_from_line(idx, node_dict, line, lines):
    type_line = lines[idx + 1]
    shape_line = lines[idx + 2]
    name_line = lines[idx + 3]
    node_type = type_line.split(":")[1].strip()
    node_name = name_line.split(":")[1].strip()
    node_full_name = name_line[name_line.index(":") + 1:].strip()
    node_shape = tuple(int(shape) if shape != '-1' else None
                       for shape in shape_line.split(":")[1].strip()[1:-1].split(","))
    node_dict[line[line.index("'") + 1:line.rfind("'")]] = NodeInfo(node_name, node_shape, node_type, node_full_name)


def saved_pb(saved_model_dir, output_nodes, output_dir, output_name, saved_tags):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.load(sess, saved_tags, saved_model_dir)
        graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=[node.name
                                                                                          for node
                                                                                          in output_nodes.values()])

        tf.io.write_graph(
            graph,
            output_dir,
            output_name,
            as_text=False
        )

        return tuple(tensor.name for tensor in graph.node)


def saved_model_to_pb(saved_model_dir, output_dir, output_name, saved_tags=tag_constants.SERVING,
                      sign=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):

    input_nodes, output_nodes = get_input_output_node(saved_model_dir, saved_tags, sign)

    print("[INFO]: Save Model has [", len(output_nodes), "] outputs.")
    print("[INFO]: Outputs Nodes: ", output_nodes, ".")

    saved_node_names = saved_pb(saved_model_dir, output_nodes, output_dir, output_name, [saved_tags])

    remove_keys = []
    for key, value in input_nodes.items():
        if value.name not in saved_node_names:
            remove_keys.append(key)

    for key in remove_keys:
        input_nodes.pop(key)
    print("[INFO]: Inputs Nodes With Shapes: ", input_nodes, ".")
    print("[INFO]: Saved Model convert to Frozen Model done.")
    return input_nodes, output_nodes


def saved_sub_graph_saved_model(new_input_nodes, new_output_nodes, saved_model_dir, tmp_path,
                                saved_tags=tag_constants.SERVING,
                                sign=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    if new_input_nodes is not None:
        input_node_names_dict = parse_new_input_nodes_string(new_input_nodes)
    else:
        input_node_names_dict = OrderedDict()

    if new_output_nodes is not None:
        output_node_names_dict = parse_new_output_nodes_string(new_output_nodes)
    else:
        _, output_node_dict = get_input_output_node(saved_model_dir, saved_tags, sign)
    input_node_names = {input_node.name: input_node for input_node in input_node_names_dict.values()}
    output_node_names = list(output_node.name for output_node in output_node_names_dict.values())
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.load(sess, [saved_tags], saved_model_dir)
        input_graph_def = sess.graph.as_graph_def()
        inputs_replaced_graph_def = graph_pb2.GraphDef()
        for node in input_graph_def.node:
            if node.name in input_node_names:
                placeholder_node = node_def_pb2.NodeDef()
                placeholder_node.op = "Placeholder"
                placeholder_node.name = node.name
                placeholder_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=getattr(types_pb2, input_node_names.get(node.name).type)))
                inputs_replaced_graph_def.node.extend([placeholder_node])
            else:
                inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])
        output_graph_def = graph_util.extract_sub_graph(inputs_replaced_graph_def, output_node_names)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(output_graph_def, name="")
        graph = tf.get_default_graph()
        labeling_signature = tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={k: tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name(v.full_name))
                            for k, v in input_node_names_dict.items()},
                    outputs={k: tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name(v.full_name))
                             for k, v in output_node_names_dict.items()})
        output_path = os.path.join(tmp_path, "tmp_saved_model")
        saved_model_builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        saved_model_builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: labeling_signature})
        saved_model_builder.save()
    return output_path


def parse_new_input_nodes_string(new_nodes_string):
    node_dict = OrderedDict()
    for new_node_string in new_nodes_string.split(";"):
        node_name_split = new_node_string.split(":")
        name = node_name_split[0].strip()
        node_type = node_name_split[1].strip()
        node_dict[node_name_split[0].strip()] = NodeInfo(name, node_type, None,
                                                         ":".join(node_name_split[2:]).strip())
    return node_dict


def parse_new_output_nodes_string(new_nodes_string):
    node_dict = OrderedDict()
    for new_node_string in new_nodes_string.split(";"):
        node_name_split = new_node_string.split(":")
        node_dict[node_name_split[0].strip()] = NodeInfo(node_name_split[1].strip(), None, None,
                                                         ":".join(node_name_split[1:]).strip())
    return node_dict


def save_hw_saved_model(input_node_dict: dict, output_node_dict: dict, output_path, method_name):
    try:
        from npu_bridge.helper import helper
    except ImportError:
        print("[ERROR]: npu_bridge is not found, HW Saved Model will not be generated.")
        return

    tf.disable_eager_execution()
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["graph_run_mode"].i = 1
    with tf.Session(config=config) as sess:
        hw_saved_model_input_dict = OrderedDict()
        for key, value in input_node_dict.items():
            hw_saved_model_input_dict[key] = tf.placeholder(shape=value.shape,
                                                            dtype=tf.DType(getattr(types_pb2, value.type)),
                                                            name=value.name)
        gen_npu_ops = helper.get_gen_ops()
        outputs = gen_npu_ops.load_and_execute_om(list(hw_saved_model_input_dict.values()),
                                                  output_dtypes=tuple(tf.DType(getattr(types_pb2, value.type))
                                                                      for value in output_node_dict.values()),
                                                  om_path=f"{output_path}.om")

        # test run pattern
        # import numpy as np
        # input_list = list(hw_saved_model_input_dict.values())
        # print(sess.run(outputs, feed_dict={input_list[0]: np.random.randn(1, 128).astype(np.int32),
        #                                    input_list[1]: np.random.randn(1, 128).astype(np.int32),
        #                                    input_list[2]: np.random.randn(1, 128).astype(np.int32)}))

        saved_outputs = OrderedDict()
        for output_tensor, output_node in zip(outputs, output_node_dict):
            saved_outputs[output_node] = tf.saved_model.utils.build_tensor_info(output_tensor)
        labeling_signature = tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={k: tf.saved_model.utils.build_tensor_info(v) for k, v in hw_saved_model_input_dict.items()},
                    outputs=saved_outputs,
                    method_name=method_name)
        saved_path = output_path + f"_load_om_saved_model_{strftime('%Y%m%d_%H_%M_%S',localtime())}"
        saved_model_builder = tf.saved_model.builder.SavedModelBuilder(saved_path)
        saved_model_builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: labeling_signature})

        saved_model_builder.save()
        print(f"[INFO]: HW Saved Model is saved to {saved_path}.")


def gen_pb_txt(pb_file_path):
    func2graph_path_in_home = os.path.join("Ascend", "ascend-toolkit", "latest", "atc", "python", "func2graph",
                                           "func2graph.py")
    func2graph_path = os.path.join(os.getenv("HOME"), func2graph_path_in_home)
    if not os.path.exists(func2graph_path):
        func2graph_path = os.path.join("/usr/local", func2graph_path_in_home)
    subprocess.run(("python3", func2graph_path, "-m", pb_file_path), stdout=subprocess.PIPE)


def pb_to_om(pb_file_path, output_path, soc_version, input_shape, out_nodes, rest_args):
    atc_path_in_home = os.path.join("Ascend", "ascend-toolkit", "latest", "bin", "atc")
    atc_path = os.path.join(os.getenv("HOME"), atc_path_in_home)
    if not os.path.exists(atc_path):
        atc_path = os.path.join("/usr/local", atc_path_in_home)
    return subprocess.run((atc_path, "--framework", "3", "--model", pb_file_path, "--output", output_path,
                           "--soc_version", soc_version, "--input_shape", input_shape, "--out_nodes", out_nodes,
                           *rest_args))


def pb_to_om_with_profiling(pb_file_path, output_path, input_shape, out_nodes, profiling, rest_args):
    aoe_path_in_home = os.path.join("Ascend", "ascend-toolkit", "latest", "bin", "aoe")
    aoe_path = os.path.join(os.getenv("HOME"), aoe_path_in_home)
    if not os.path.exists(aoe_path):
        aoe_path = os.path.join("/usr/local", aoe_path_in_home)
    return subprocess.run((aoe_path, "--framework", "3", "--model", pb_file_path, "--output",
                           output_path, "--input_shape", input_shape, "--out_nodes", out_nodes,
                           "--job_type", profiling, *rest_args))


def get_input_shape(input_nodes):
    input_shapes = []
    for value in input_nodes.values():
        input_shapes.append(f"{value.name}:{','.join('1' if shape is None else str(shape) for shape in value.shape)}")
    input_shape_param = ";".join(input_shapes)
    print(f"Because the input_shape parameter is not set, the following values are used: {input_shape_param}")
    return input_shape_param


def get_out_nodes(output_nodes):
    return ";".join(output_node.full_name for output_node in output_nodes.values())


def main(input_path, output_path, input_shape, soc_version, profiling, method_name, new_input_nodes, new_output_nodes,
         rest_args):
    try:
        tmp_path = os.path.join("/tmp", f"pb_dir_{int(time.time())}")
        os.makedirs(tmp_path, exist_ok=True)
        tmp_pb_name = "model.pb"
        tmp_pb_file = os.path.join(tmp_path, tmp_pb_name)
        if new_input_nodes is not None or new_output_nodes is not None:
            input_path = saved_sub_graph_saved_model(new_input_nodes, new_output_nodes, input_path, tmp_path)
        input_nodes, output_nodes = saved_model_to_pb(input_path, tmp_path, tmp_pb_name)
        if not input_shape:
            input_shape = get_input_shape(input_nodes)
        gen_pb_txt(tmp_pb_file)
        out_nodes = get_out_nodes(output_nodes)
        if profiling is not None:
            ret = pb_to_om_with_profiling(tmp_pb_file, output_path, input_shape, out_nodes, profiling, rest_args)
        else:
            ret = pb_to_om(tmp_pb_file, output_path, soc_version, input_shape, out_nodes, rest_args)
        if ret.returncode != 0:
            return
        print(f"[INFO]: The om model has been converted and the HW Saved Model is ready to be generated.")
        save_hw_saved_model(input_nodes, output_nodes, output_path, method_name)
    finally:
        shutil.rmtree(tmp_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help='SavedModel path.')
    parser.add_argument("--output_path", required=True,
                        help="Output file path&name(needn't suffix, will add .om automatically).")
    parser.add_argument("--input_shape", default='', help='Shape of input data. '
                                                          'Separate multiple nodes with semicolons (;). '
                                                          'Use double quotation marks (") to enclose each argument.'
                                                          'E.g.: "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"')
    parser.add_argument("--method_name", help='Method name for TF-Serving.')
    # parser.add_argument("--new_input_nodes",
    #                     help='Configure this to reselect the input node.'
    #                          'the node format is name:type_pb:node_name'
    #                          'Separate multiple nodes with semicolons (;).'
    #                          'Use double quotation marks (") to enclose each argument.'
    #                          'E.g.: "embedding:DT_FLOAT:bert/embedding/word_embeddings:0;add:DT_INT:bert/embedding/add:0"')
    # parser.add_argument("--new_output_nodes",
    #                     help='Configure this to reselect the output node.'
    #                     'Separate multiple nodes with semicolons (;).'
    #                     'Use double quotation marks (") to enclose each argument.'
    #                     'E.g.: "loss:loss/Softmax:0"')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--soc_version", help="The soc version. "
                                             "This parameter is not required when profiling is set.")
    group.add_argument("--profiling", choices=['1', '2'], help="Set this parameter when profiling is required."
                                                               "(sgat: 1, opat: 2).")
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown_args = get_args()
    main(args.input_path, args.output_path, args.input_shape, args.soc_version, args.profiling, args.method_name,
         None, None, unknown_args)