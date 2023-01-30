import sys
import numpy as np
import torch

from ais_bench.infer.interface import InferSession

model_path = sys.argv[1]

# 最短运行样例
def infer_simple():
    device_id = 0
    session = InferSession(device_id, model_path)

    # create npy array
    ndata = np.zeros([1,256,256,3], dtype=np.uint8)

    # in is numpy list and ouput is numpy list
    outputs = session.infer([ndata])
    print("in npy outputs[0].shape:{} type:{}".format(outputs[0].shape, type(outputs)))


    # create torch tensor
    torchtensor = torch.tensor(ndata)
    # in is torch tensor and ouput is numpy list
    outputs = session.infer([torchtensor])
    print("in torch tensor outputs[0].shape:{} type:{}".format(outputs[0].shape, type(outputs)))


    # create discontinuous torch tensor
    ndata = np.zeros([1,256,256,3], dtype=np.uint8)
    torchtensor = torch.tensor(ndata, dtype=torch.uint8)
    torchtensor_transposition = torchtensor.permute(0,2,1,3)

    # in is discontinuous tensor list and ouput is numpy list
    outputs = session.infer([torchtensor_transposition])
    print("in discontinuous torch tensor outputs[0].shape:{} type:{}".format(outputs[0].shape, type(outputs)))

    print("static infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))

def infer_dymshape():
    device_id = 0
    session = InferSession(device_id, model_path)

    ndata = np.zeros([1,3,224,224], dtype=np.float32)

    mode = "dymshape"
    # input args custom_sizes is int
    outputSize = 100000
    outputs = session.infer([ndata], mode, custom_sizes=outputSize)
    print("inputs: custom_sizes: {} outputs:{} type:{}".format(outputSize, outputs, type(outputs)))

    # input args custom_sizes is list
    outputSizes = [100000]
    outputs = session.infer([ndata], mode, custom_sizes=outputSizes)
    print("inputs: custom_sizes: {} outputs:{} type:{}".format(outputSizes, outputs, type(outputs)))

    ndata = torch.rand([1,224,3,224], out=None, dtype=torch.float32)
    ndata1 = ndata.permute(0,2,1,3)
    ndata11 = ndata1.contiguous()

    outputs = session.infer([ndata11], mode, custom_sizes=outputSize)
    print("inputs: custom_sizes: {} outputs:{} type:{} tesnor is_contiguous: {}".format(outputSize, outputs, type(outputs), ndata11.is_contiguous()))

    print("dymshape infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))

def infer_dymdims():
    device_id = 0
    session = InferSession(device_id, model_path)

    ndata = np.zeros([1,3,224,224], dtype=np.float32)

    mode = "dymdims"
    outputs = session.infer([ndata], mode)
    print("outputs:{} type:{}".format(outputs, type(outputs)))

    print("dymdims infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))

# 获取模型信息
def get_model_info():
    device_id = 0
    session = InferSession(device_id, model_path)

    # 方法2 直接打印session 也可以获取模型信息
    print(session.session)

    # 方法3 也可以直接通过get接口去获取
    intensors_desc = session.get_inputs()
    for i, info in enumerate(intensors_desc):
        print("input info i:{} shape:{} type:{} val:{} realsize:{} size:{}".format(
            i, info.shape, info.datatype, int(info.datatype), info.realsize, info.size))

    intensors_desc = session.get_outputs()
    for i, info in enumerate(intensors_desc):
        print("outputs info i:{} shape:{} type:{} val:{} realsize:{} size:{}".format(
            i, info.shape, info.datatype, int(info.datatype), info.realsize, info.size))

infer_simple()
#infer_dymshape()
# infer_dymdims()
#get_model_info()
