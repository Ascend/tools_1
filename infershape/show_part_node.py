import onnx
import os 
import sys

from google.protobuf import text_format

def load_groph_def_from_pb(path):
    with open(path, "rb") as f:
        data = f.read()
        model = onnx.ModelProto()
        text_format.Parse(data, model)
    return model.graph
    
def get_node_shape(graph, node_name):
    shape_attr = None
    for node in graph.node:
        if str(node.name) == str(node_name):
            for attr in node.attribute:
                if (attr.name == "output_desc_shape:0"):
                    shape_attr = attr
            if (shape_attr == None):
                return []
            return "[" + ",".join('%s' %id for id in shape_attr.ints) + "]" + r'\n' + 'type:' + str(shape_attr.type)

def bfsTravel(graph, source, show_level):
    frontiers = [source]
    travel = [source]
    while show_level:
        nexts = []
        for frontier in frontiers:
            for node in graph.node:
                if str(node.name) == str(frontier).replace(':0',''):
                    input_len = len(node.input)
                    while input_len:
                        input_len -= 1
                        travel.append(node.input[input_len])
                        nexts.append(node.input[input_len])
        frontiers = nexts
        show_level -= 1
    return travel

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("leak argument")
        print("argument1 : onnx graph file name(can reach)")
        print("argument2 : target onde(full name)")
        print("argument3 : back layer from target node")
        exit(0)
    total_content = 'digraph G {\nrankdir = "TB";\nnode[shape = "box", with = 0, height = 0];\nedge[arrowhead = "none", style = "solid"];\n'
    with open("part_node.dot", "w") as f:
        f.write(total_content)
        
    graph_def = load_graph_def_from_pb(sys.argv[1])
    target_node = sys.argv[2]
    show_level = int(sys.argv[3]) - 1
    
    for tnode in bfsTravel(graph_def, target_node, show_level):
        for node in graph_def.node:
            if str(node.name) == str(tnode).replace(':0',''):
                input_len = len(node.input)
                while input_len:
                    input_len -= 1
                    if len(node.input[input_len]) != 0:
                        shape_content = '"[label="shape:' + get_node_shape(graph_def, node.input[input_len].replace(':0',''))
                        total_content = '"' + node.input[input_len].replace(':0','').replace("/",r"\n/") + '" -> "' + node.name.replace("/",r"\n/") + shape_content + '", arrowhead="normal"];\n'
                        with open("part_node.dot", "a+") as f:
                            f.write(total_content)
    with open("part_node.dot", "a+") as f:
        f.write("}")
        
    commend = "dot -T png -o part_node.png part_node.dot"
    os.system(commend)
    
    