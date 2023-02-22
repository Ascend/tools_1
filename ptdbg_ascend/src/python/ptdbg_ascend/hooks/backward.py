import torch
from ..common.utils import print_warn_log


class Backward:
    def __init__(self):
        self.tensors = None
        self.gradient = None
        self.retain_graph = None
        self.create_graph = False
        self.inputs = None

    @staticmethod
    def _tensor_or_tensors_to_tuple(tensors, length):
        if tensors is None:
            return (None, ) * length
        if isinstance(tensors, torch.Tensor):
            return (tensors, )
        return tuple(tensors)

    @staticmethod
    def _make_grads(outputs, grads):
        new_grads = []
        for out, grad in zip(outputs, grads):
            if isinstance(grad, torch.Tensor):
                if not out.shape == grad.shape:
                    raise RuntimeError("Mismatch in shape: grad_output["
                                       + str(grads.index(grad)) + "] has a shape of "
                                       + str(grad.shape) + " and output["
                                       + str(outputs.index(out)) + "] has a shape of "
                                       + str(out.shape) + ".")
                if out.dtype.is_complex != grad.dtype.is_complex:
                    raise RuntimeError("For complex Tensors, both grad_output and output"
                                       " are required to have the same dtype."
                                       " Mismatch in dtype: grad_output["
                                       + str(grads.index(grad)) + "] has a dtype of "
                                       + str(grad.dtype) + " and output["
                                       + str(outputs.index(out)) + "] has a dtype of "
                                       + str(out.dtype) + ".")
                new_grads.append(grad)
            elif grad is None:
                if out.requires_grad:
                    if out.numel() != 1:
                        raise RuntimeError("grad can be implicitly created only for scalar outputs")
                    new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
                else:
                    new_grads.append(None)
            else:
                raise TypeError("gradients can be either Tensors or None, but got " +
                                type(grad).__name__)
        return tuple(new_grads)

    def backward(self, tensors, grad_tensors=None, retain_graph=None, create_graph=False,
                 grad_variables=None, inputs=None):
        from torch.autograd import Variable
        if tensors == None:
            return
        self.tensors = tensors
        self.gradient = grad_tensors
        self.retain_graph = retain_graph
        self.create_graph = create_graph
        self.inputs = inputs
        if grad_variables is not None:
            print_warn_log("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
            if grad_tensors is None:
                grad_tensors = grad_variables
            else:
                raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) "
                                   "arguments both passed to backward(). Please only "
                                   "use 'grad_tensors'.")
        if inputs is not None and len(inputs) == 0:
            raise RuntimeError("'inputs' argument to backward() cannot be empty.")

        tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
        inputs = tuple(inputs) if inputs is not None else tuple()

        grad_tensors_ = self._tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
        grad_tensors_ = self._make_grads(tensors, grad_tensors_)
        if retain_graph is None:
            retain_graph = create_graph

        Variable._execution_engine.run_backward(
            tensors, grad_tensors_, retain_graph, create_graph, inputs,
            allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag