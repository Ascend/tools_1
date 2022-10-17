
import time
import aclruntime
from frontend.summary import summary

class InferSession:
    def __init__(self, device_id: int, model_path: str, acl_json_path: str = None, debug: bool = False, loop: int = 1):
        self.device_id = device_id
        self.model_path = model_path
        self.acl_json_path = acl_json_path
        self.log_level = 1 if debug == True else 0
        self.loop = loop
        options = aclruntime.session_options()
        self.session = aclruntime.InferenceSession(self.model_path, self.device_id, options)
        self.outputs_names = [meta.name for meta in self.session.get_outputs()]    

    def get_inputs(self):
        self.intensors_desc = self.session.get_inputs()
        return self.intensors_desc

    def get_outputs(self):
        self.outtensors_desc = self.session.get_outputs()
        return self.outtensors_desc

    # 默认设置为静态batch
    def set_staticbatch(self):
        self.session.set_staticbatch()

    def set_dynamic_batchsize(self, dymBatch: str):
        self.session.set_dynamic_batchsize(dymBatch)

    def set_dynamic_hw(self, w: int, h: int):
        self.session.set_dynamic_hw(w, h)

    def set_dynamic_dims(self, dym_dims: str):
        self.session.set_dynamic_dims(dym_dims)

    def set_dynamic_shape(self, dym_shape: str):
        self.session.set_dynamic_shape(dym_shape)

    def set_custom_outsize(self, custom_sizes):
        self.session.set_custom_outsize(custom_sizes)

    def create_tensor_from_numpy_to_device(self, ndata):
        tensor = aclruntime.Tensor(ndata)
        starttime = time.time()
        tensor.to_device(self.device_id)
        endtime = time.time()
        summary.h2d_latency_list.append(float(endtime - starttime) * 1000.0)  # millisecond
        return tensor

    def run(self, feeds):
        outputs = self.session.run(self.outputs_names, feeds)
        return outputs

    def convert_tensors_to_host(self, tensors):
        totle_laency = 0.0
        for i, out in enumerate(tensors):
            starttime = time.time()
            out.to_host()
            endtime = time.time()
            totle_laency += float(endtime - starttime) * 1000.0  # millisecond
        summary.d2h_latency_list.append(totle_laency)

    def reset_sumaryinfo(self):
        self.session.reset_sumaryinfo()

    def sumary(self):
        return self.session.sumary()